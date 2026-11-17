# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive failure explorer — expand functions → buckets → examples.

Arrow keys navigate; Enter/→ expands; ←/Backspace collapses; Esc/q quits.
Per-bucket row shows PRECISION (fires → % fail) like dashboard_cli / v7.
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from autofa.behavior_policy import strictness_for_config
from autofa.core.library import FunctionLibrary
from autofa.metrics.verdicts import (
    extract_metrics,
    format_behavior_purity,
    format_interval,
    format_metrics_line,
    format_task_recall,
    is_sampled,
    sampled_marker,
    verdict_for_scores,
    verdict_legend,
    verdict_style,
)
from autofa.reports.ordering import rank_display_functions
from autofa.ui.errors import show_action_error, show_action_notice
from autofa.ui.pt_app import run_app
from autofa.ui.syntax import highlight_python_lines
from autofa.ui.theme import MENU_STYLE


def _mechanism_prose(fn: dict[str, Any]) -> str:
    """Narrative of what failure this detector tags (HTML dashboard 'Mechanism')."""
    mech = (fn.get("mechanism") or "").strip()
    desc = (fn.get("description") or "").strip()
    if mech and desc and mech != desc:
        return mech
    return mech or desc


def _wrap_prose(text: str, *, width: int = 92, indent: str = "") -> list[str]:
    import textwrap

    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        if not para.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                para.strip(),
                width=width,
                subsequent_indent="",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [para.strip()]
        )
    return [f"{indent}{ln}" if ln else indent.rstrip() for ln in lines]


def _purity_bar(purity: float, *, width: int = 24, high: bool = False) -> str:
    """ASCII precision bar: filled = % fail among fires."""
    p = max(0.0, min(1.0, float(purity)))
    filled = int(round(p * width))
    empty = width - filled
    # Green for high-purity (≥70%), dim otherwise — matches dashboard_v7 hp wash.
    if high or p >= 0.70:
        return "█" * filled + "░" * empty
    if p >= 0.40:
        return "▓" * filled + "░" * empty
    return "▒" * filled + "░" * empty


def _buckets_for(fn: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = (fn.get("scores") or {}).get("buckets")
    return buckets if isinstance(buckets, list) else []


@dataclass
class _Node:
    kind: str  # fn | bucket | example | meta
    label: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    children: list[_Node] = field(default_factory=list)
    expanded: bool = False


def _build_tree(library: FunctionLibrary, *, top_x: int | None = None,
                objective_config=None) -> list[_Node]:
    ranked = rank_display_functions(
        [vars(fn) for fn in library.functions],
        behavior_strictness=(strictness_for_config(objective_config)
                             if objective_config is not None else None),
        review_history=library.review_history,
    )[:top_x or None]
    roots: list[_Node] = []
    for i, fn in enumerate(ranked, 1):
        scores = fn["scores"]
        m = extract_metrics(scores)
        v = verdict_for_scores(scores)
        name = fn.get("name") or "unnamed"
        badge = f"{v['symbol']} {v['label']}  "
        title = f"{badge}#{i} {name}"
        if (fn.get("ranking") or {}).get("eligible") is False:
            title += " · Research candidate"
        if m["evaluation_status"] == "unscored":
            summary = v["reason"]
        else:
            recall_txt = (f"{m['recall']:.2f}{format_interval(scores, 'recall')}"
                          if m["coverage_available"] else "—")
            ppv_txt = (f"{m['ppv_at_pi0']:.2f}"
                       if m["normalized_precision_available"] else "—")
            precision_txt = f"{m['precision']:.2f}" if m["precision"] is not None else "—"
            summary = (
                f"{v['reason']}  ·  "
                f"%caught={m['flagged_detail']}  "
                f"precision={precision_txt}  "
                f"attempt_recall={recall_txt}  "
                f"task_recall={format_task_recall(scores)}  "
                f"ppv@π₀={ppv_txt}  "
                f"{format_behavior_purity(scores).replace('beh=', 'behavior_purity=')}"
                f"{sampled_marker(scores, short=True)}"
                f" · {m['scoring_detail']}"
                + (f" · {m['diagnostic_review_detail']}"
                   if v["code"] != "UNSCORED" and m["diagnostic_review_detail"] else "")
            )
        fn_node = _Node(
            kind="fn",
            label=title,
            detail=summary,
            data={**fn, "scores": scores, "_verdict": v},
        )

        # Mechanism first (HTML dashboard labelled field) — orients the reader.
        prose = _mechanism_prose(fn)
        if prose:
            fn_node.children.append(
                _Node(
                    kind="mechanism",
                    label="Mechanism",
                    detail=prose,
                    data={"text": prose},
                    expanded=True,
                )
            )
        how = (fn.get("function_mechanism") or "").strip()
        if not how:
            from autofa.reports.detector_summary import summarize_function_mechanism

            how = summarize_function_mechanism(
                fn.get("source") or "", kind=str(fn.get("kind") or "regex")
            )
        if how:
            fn_node.children.append(
                _Node(
                    kind="mechanism",
                    label="How the detector works",
                    detail=how,
                    data={"text": how},
                    expanded=True,
                )
            )
        desc = (fn.get("description") or "").strip()
        mech = (fn.get("mechanism") or "").strip()
        if desc and mech and desc != mech:
            fn_node.children.append(
                _Node(
                    kind="meta",
                    label="title",
                    detail=desc,
                    data={"text": desc},
                )
            )

        # Metrics meta with legend
        fn_node.children.append(
            _Node(
                kind="meta",
                label="metrics",
                detail=format_metrics_line(scores),
                data={},
            )
        )
        fn_node.children.append(
            _Node(
                kind="meta",
                label="verdict",
                detail=f"{v['symbol']} {v['label']} — {v['reason']}",
                data=v,
            )
        )
        if is_sampled(scores):
            fn_node.children.append(
                _Node(
                    kind="meta",
                    label="scored on",
                    detail=(
                        f"stratified sample of {scores.get('sample_n')} of "
                        f"{scores.get('sample_frame_n')} rows across "
                        f"{scores.get('sample_strata')} strata (LLM detector) — "
                        f"recall 95% CI {format_interval(scores, 'recall') or 'n/a'}, "
                        f"precision 95% CI {format_interval(scores, 'raw_ppv') or 'n/a'}"
                    ),
                    data={},
                )
            )

        buckets = _buckets_for(fn)
        bucket_parent = _Node(
            kind="meta",
            label=f"buckets ({len(buckets)})",
            detail="PRECISION (fires → % fail)",
            data={},
            expanded=True,
        )
        for b in buckets:
            purity = float(b.get("purity") or 0)
            high = bool(b.get("is_high_purity")) or purity >= 0.70
            bar = _purity_bar(purity, high=high)
            tag = "● HIGH" if high else ("✗ LOW" if purity < 0.5 else "◐ MID")
            detail = f"{tag}  {bar}  precision={purity:.0%} fail"
            extra = (
                f"  n={b.get('n')} n_fail={b.get('n_fail')}  "
                f"rec={float(b.get('recall') or 0):.2f}  "
                f"ppv@π₀={float(b.get('ppv_at_pi0') or 0):.2f}"
            )
            b_node = _Node(
                kind="bucket",
                label=str(b.get("bucket")),
                detail=detail + extra,
                data=b,
            )
            examples = b.get("example_fail_ids") or []
            if examples:
                for eid in examples:
                    b_node.children.append(
                        _Node(kind="example", label=f"fail id  {eid}", data={"id": eid})
                    )
            elif (fn.get("example_fail_ids")
                  and b.get("bucket") == (scores.get("bucket") or "positive")):
                for eid in (fn.get("example_fail_ids") or [])[:5]:
                    b_node.children.append(
                        _Node(kind="example", label=f"fail id  {eid}", data={"id": eid})
                    )
            bucket_parent.children.append(b_node)
        fn_node.children.append(bucket_parent)

        src = (fn.get("source") or "").strip()
        if src:
            n_lines = len(src.splitlines())
            fn_node.children.append(
                _Node(
                    kind="source",
                    label=f"source  ({n_lines} lines)  ·  →/Enter open",
                    detail="",
                    data={"source": src},
                )
            )

        roots.append(fn_node)
    if library.helpers.aliases:
        helpers_node = _Node(kind="helpers", label="Reusable helpers (unscored)")
        for alias, ref in sorted(library.helpers.aliases.items()):
            helper = library.helpers.revisions[ref]
            helpers_node.children.append(_Node(
                kind="source", label=f"{alias} · {helper.purpose} · {ref[:12]}",
                detail=helper.description, data={"source": helper.source},
            ))
        roots.append(helpers_node)
    return roots


def _visible(nodes: list[_Node], depth: int = 0) -> list[tuple[_Node, int]]:
    out: list[tuple[_Node, int]] = []
    for n in nodes:
        out.append((n, depth))
        if n.expanded and n.children:
            out.extend(_visible(n.children, depth + 1))
    return out


def ensure_line_scroll(
    scroll: int,
    *,
    sel_start: int,
    sel_end: int,
    n_lines: int,
    capacity: int,
) -> int:
    """Keep the selected line range overlapping ``[scroll, scroll+capacity)``.

    Tall selections (expanded source) allow scrolling inside the selection;
    the viewport is only clamped so it still overlaps that range.
    """
    if n_lines <= 0 or capacity <= 0:
        return 0
    if n_lines <= capacity:
        return 0
    sel_start = max(0, min(sel_start, n_lines - 1))
    sel_end = max(sel_start + 1, min(sel_end, n_lines))
    height = sel_end - sel_start
    if height >= capacity:
        min_scroll = sel_start
        max_scroll = max(sel_start, sel_end - capacity)
        if scroll < min_scroll:
            scroll = min_scroll
        elif scroll > max_scroll:
            scroll = max_scroll
    else:
        if sel_start < scroll:
            scroll = sel_start
        elif sel_end > scroll + capacity:
            scroll = sel_end - capacity
        if sel_start >= scroll + capacity:
            scroll = sel_start
    return max(0, min(scroll, n_lines - capacity))


def _flatten_visual_lines(
    visual: list[list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in visual:
        out.extend(row)
        if not row or not row[-1][1].endswith("\n"):
            out.append(("", "\n"))
    return out


def explore_library(  # noqa: C901
    library: FunctionLibrary,
    *,
    top_x: int | None = None,
    title: str | None = None,
    html_path: Path | None = None,
    dataset: object | None = None,
    source_harness: str | None = None,
    reference_data: list[dict[str, str]] | None = None,
) -> None:
    """Interactive arrow-key explorer for a function library.

    Also writes / refreshes ``failure_cards.html`` and shows its URL in the
    header (press ``o`` to open in the browser). The TUI replaces earlier
    Rich output, so the link lives here — not only before you enter.

    When ``dataset`` is provided (or a run ``config.json`` sits next to the
    HTML output), example fail trajectories are embedded for click-to-inspect.
    """
    from rich.console import Console

    from autofa.ui.menus import terminal_rows

    console = Console()
    if not library.functions:
        show_action_notice("No detectors in library yet. Start or resume discovery first.",
                           console=console)
        return

    html_url = ""
    try:
        from autofa.config import autofa_home
        from autofa.reports.failure_cards import (
            REPORT_NAME,
            file_url,
            serve_failure_cards,
            write_failure_cards_with_status,
        )

        out = Path(html_path) if html_path is not None else (
            autofa_home() / "cache" / REPORT_NAME
        )
        out = write_failure_cards_with_status(
            library,
            out=out,
            dataset=dataset,
            source_harness=source_harness,
            reference_data=reference_data,
            console=console,
        )
        try:
            from autofa.reports.efficiency_card import attach_saved_efficiency

            attach_saved_efficiency(out, library)
        except Exception as embed_error:  # noqa: BLE001
            show_action_notice(
                f"Efficiency metrics were not added: {embed_error}",
                console=console,
            )
        html_path = out
        # Prefer localhost so full-text sidecars can be fetched on click.
        try:
            html_url = serve_failure_cards(out)
        except Exception:  # noqa: BLE001
            html_url = file_url(out)
    except Exception as e:  # noqa: BLE001
        show_action_error(e, console=console, context="Generate HTML report")

    # Drop library-browser / status-spinner chrome so the TUI is the only viewport.
    try:
        from autofa.ui.menus import prepare_step

        prepare_step()
    except Exception:  # noqa: BLE001
        pass

    from autofa.objectives import config_from_run_dir

    roots = _build_tree(library, top_x=top_x, objective_config=config_from_run_dir(
        Path(html_path).parent if html_path is not None else None,
    ))
    # Expand first function by default so buckets are visible immediately.
    if roots:
        roots[0].expanded = True

    state = {"index": 0, "scroll": 0, "quit": False}
    hdr = title or f"AutoFA explorer · {library.model_id} / {library.benchmark or 'all'}"

    def _rows() -> list[tuple[_Node, int]]:
        return _visible(roots)

    def _header_lines() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [
            ("class:title", f"  {hdr}\n"),
            ("class:hint", "  "),
            ("class:key", "↑/↓"),
            ("class:hint", " move · "),
            ("class:key", "PgUp/PgDn"),
            ("class:hint", " scroll · "),
            ("class:key", "→/Enter"),
            ("class:hint", " expand · "),
            ("class:key", "←"),
            ("class:hint", " collapse · "),
            ("class:key", "Esc/q"),
            ("class:hint", " quit"),
        ]
        if html_url:
            lines.extend(
                [("class:hint", " · "), ("class:key", "o"), ("class:hint", " open HTML")]
            )
        lines.extend(
            [
                ("class:hint", "\n"),
                (
                    "class:hint",
                    f"  {verdict_legend()}\n",
                ),
                (
                    "class:hint",
                    "  Mechanism = why the model fails (actionable story)"
                    " · buckets = fires → % fail\n",
                ),
            ]
        )
        if html_url:
            lines.extend(
                [
                    ("class:accent", "  HTML failure cards  (press "),
                    ("class:key", "o"),
                    ("class:accent", ")\n"),
                ]
            )
            lines.append(("class:dim", f"  {html_url}\n"))
        lines.append(("", "\n"))
        return lines

    def _append_row(
        visual: list[list[tuple[str, str]]], *parts: tuple[str, str]
    ) -> None:
        visual.append(list(parts))

    def _content_visual() -> tuple[  # noqa: C901
        list[list[tuple[str, str]]],
        list[tuple[int, int]],
    ]:
        """Visual rows (one screen line each) plus range per tree node."""
        rows = _rows()
        if state["index"] >= len(rows):
            state["index"] = max(0, len(rows) - 1)
        visual: list[list[tuple[str, str]]] = []
        ranges: list[tuple[int, int]] = []
        for i, (node, depth) in enumerate(rows):
            start = len(visual)
            selected = i == state["index"]
            if selected:
                style = "class:selected"
            elif node.kind == "fn":
                vcode = (node.data.get("_verdict") or {}).get("code") or "OK"
                style = f"class:{verdict_style(vcode)}"
            elif node.kind == "mechanism":
                style = "class:accent"
            else:
                style = "class:item"
            indent = "  " * (depth + 1)
            has_kids = bool(node.children)
            expandable = has_kids or node.kind == "source"
            if node.kind == "fn":
                marker = "▼ " if node.expanded else ("▶ " if expandable else "  ")
            elif node.kind == "bucket":
                marker = (
                    "▼ " if node.expanded and has_kids else ("▸ " if has_kids else "· ")
                )
            elif node.kind == "source":
                marker = "▼ " if node.expanded else "▶ "
            elif node.kind == "mechanism":
                marker = "  "
            else:
                marker = "  "
            prefix = "❯ " if selected else "  "
            if node.kind == "source" and "→/Enter" in node.label:
                before, after = node.label.split("→/Enter", 1)
                key_style = style if selected else "class:key"
                _append_row(
                    visual,
                    (style, f"{prefix}{indent}{marker}{before}"),
                    (key_style, "→/Enter"),
                    (style, after),
                )
            else:
                _append_row(visual, (style, f"{prefix}{indent}{marker}{node.label}"))

            if node.kind == "mechanism" and node.detail:
                prose_indent = " " * (len(prefix) + len(indent) + 2)
                dstyle = "class:selected" if selected else "class:item"
                for wrapped in _wrap_prose(node.detail, width=88, indent=prose_indent):
                    _append_row(visual, (dstyle, wrapped))
                _append_row(visual, ("", ""))
            elif node.detail and node.kind not in {"source"}:
                dstyle = "class:selected" if selected else "class:dim"
                if node.kind == "bucket":
                    _append_row(
                        visual,
                        (dstyle, f"{' ' * (len(prefix) + len(indent) + 2)}{node.detail}"),
                    )
                elif node.kind == "meta" and len(node.detail) > 100:
                    prose_indent = " " * (len(prefix) + len(indent) + 2)
                    for wrapped in _wrap_prose(node.detail, width=88, indent=prose_indent):
                        _append_row(visual, (dstyle, wrapped))
                else:
                    _append_row(visual, (dstyle, f"  {node.detail}"))

            if node.kind == "source" and node.expanded:
                src = str(node.data.get("source") or "")
                code_indent = " " * (len(prefix) + len(indent) + 2)
                # highlight_python_lines returns fragments with embedded \\n —
                # split into one visual row per source line.
                frags = highlight_python_lines(
                    src, indent=code_indent, line_numbers=True
                )
                current: list[tuple[str, str]] = []
                for style_s, text in frags:
                    if text == "\n":
                        visual.append(current)
                        current = []
                        continue
                    if "\n" in text:
                        parts = text.split("\n")
                        for j, part in enumerate(parts):
                            if j > 0:
                                visual.append(current)
                                current = []
                            if part:
                                current.append((style_s, part))
                    else:
                        current.append((style_s, text))
                if current:
                    visual.append(current)
                _append_row(visual, ("", ""))
            ranges.append((start, len(visual)))
        if not rows:
            _append_row(visual, ("class:dim", "  (empty)"))
            ranges = []
        return visual, ranges

    def _viewport_capacity() -> int:
        header_reserve = 8 if html_url else 6
        return max(5, terminal_rows() - header_reserve)

    def _sync_scroll(
        visual: list[list[tuple[str, str]]], ranges: list[tuple[int, int]]
    ) -> None:
        if not ranges:
            state["scroll"] = 0
            return
        idx = max(0, min(int(state["index"]), len(ranges) - 1))
        sel_start, sel_end = ranges[idx]
        state["scroll"] = ensure_line_scroll(
            int(state["scroll"]),
            sel_start=sel_start,
            sel_end=sel_end,
            n_lines=len(visual),
            capacity=_viewport_capacity(),
        )

    def _render() -> list[tuple[str, str]]:
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        cap = _viewport_capacity()
        scroll = int(state["scroll"])
        end = min(len(visual), scroll + cap)
        out = _header_lines()
        if scroll > 0:
            out.append(("class:dim", f"  … {scroll} lines above\n"))
        out.extend(_flatten_visual_lines(visual[scroll:end]))
        if end < len(visual):
            out.append(("class:dim", f"  … {len(visual) - end} lines below\n"))
        if len(visual) > cap:
            out.append(
                (
                    "class:dim",
                    f"\n  (row {state['index'] + 1}/{max(1, len(ranges))} · "
                    f"lines {scroll + 1}–{end}/{len(visual)})\n",
                )
            )
        return out

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:  # noqa: ANN001
        rows = _rows()
        if not rows:
            return
        visual, ranges = _content_visual()
        scroll = int(state["scroll"])
        # Prefer scrolling within tall content before moving / wrapping.
        if scroll > 0 and (
            state["index"] == 0
            or (ranges and scroll > ranges[state["index"]][0])
        ):
            # If the selection starts above the viewport top, or we're at row 0
            # with room above — nudge scroll up first when selection is tall.
            sel_start = ranges[state["index"]][0] if ranges else 0
            if scroll > sel_start or state["index"] == 0:
                state["scroll"] = max(0, scroll - 1)
                event.app.invalidate()
                return
        if state["index"] == 0:
            state["index"] = len(rows) - 1
        else:
            state["index"] -= 1
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:  # noqa: ANN001
        rows = _rows()
        if not rows:
            return
        visual, ranges = _content_visual()
        cap = _viewport_capacity()
        scroll = int(state["scroll"])
        sel_end = ranges[state["index"]][1] if ranges else 0
        # More of the selected body (or trailing content) below → scroll first.
        if scroll + cap < max(len(visual), sel_end) and (
            state["index"] >= len(rows) - 1 or scroll + cap < sel_end
        ):
            state["scroll"] = min(scroll + 1, max(0, len(visual) - cap))
            event.app.invalidate()
            return
        if state["index"] >= len(rows) - 1:
            state["index"] = 0
        else:
            state["index"] += 1
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    @kb.add("pageup")
    def _pageup(event) -> None:  # noqa: ANN001
        cap = _viewport_capacity()
        visual, ranges = _content_visual()
        state["scroll"] = max(0, int(state["scroll"]) - cap)
        if ranges:
            for i, (_start, end) in enumerate(ranges):
                if end > state["scroll"]:
                    state["index"] = i
                    break
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    @kb.add("pagedown")
    def _pagedown(event) -> None:  # noqa: ANN001
        cap = _viewport_capacity()
        visual, ranges = _content_visual()
        max_scroll = max(0, len(visual) - cap)
        state["scroll"] = min(max_scroll, int(state["scroll"]) + cap)
        if ranges:
            for i in range(len(ranges) - 1, -1, -1):
                start, _end = ranges[i]
                if start < state["scroll"] + cap:
                    state["index"] = i
                    break
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    def _can_expand(node: _Node) -> bool:
        return bool(node.children) or node.kind == "source"

    @kb.add("right")
    @kb.add("enter")
    @kb.add("l")
    def _expand(event) -> None:  # noqa: ANN001
        rows = _rows()
        if not rows:
            return
        node, _ = rows[state["index"]]
        if not _can_expand(node):
            event.app.invalidate()
            return
        node.expanded = not node.expanded
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    @kb.add("left")
    @kb.add("backspace")
    @kb.add("h")
    def _collapse(event) -> None:  # noqa: ANN001
        rows = _rows()
        if not rows:
            return
        node, depth = rows[state["index"]]
        if node.expanded and _can_expand(node):
            node.expanded = False
        elif depth > 0:
            for i in range(state["index"] - 1, -1, -1):
                if rows[i][1] < depth:
                    state["index"] = i
                    rows[i][0].expanded = False
                    break
        visual, ranges = _content_visual()
        _sync_scroll(visual, ranges)
        event.app.invalidate()

    @kb.add("o")
    def _open_html(event) -> None:  # noqa: ANN001
        if html_url:
            webbrowser.open(html_url)

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _quit(event) -> None:  # noqa: ANN001
        state["quit"] = True
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_render), wrap_lines=True)])),
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=False,
    )
    run_app(app)

    # Re-print clickable link after the TUI exits (Rich OSC-8 hyperlink).
    if html_url:
        from autofa.ui.theme import GREEN

        console.print()
        console.print(f"[{GREEN}]HTML failure cards[/]  [link={html_url}]{html_url}[/link]")
        if html_path is not None:
            console.print(f"[dim]Saved at {html_path}[/dim]")
