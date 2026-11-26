# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autofa preflight cheating`: screen a run's rollouts for cheating and write a report."""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
import time
import webbrowser
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from autofa.config import Config
from autofa.data.cohort import (
    add_cohort_arguments,
    add_taxonomy_filter_arguments,
    cohort_options,
    refine_cohort,
    resolve_inputs,
)
from autofa.data.parquet_store import Dataset
from autofa.data.remote_api import DataApiError
from autofa.preflight import PreflightResult

SOURCE_FLAGS = "--api-run, --input, --model, --benchmark, --runs-where or --latest"
CUSTOM = -1
_REVIEW_CHOICES = ((10, "10 random flagged rollouts"), (50, "50 random flagged rollouts"),
                   (0, "All flagged rollouts"), (CUSTOM, "Custom number…"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autofa preflight cheating",
        description=(
            "Screen every rollout, passed or failed, with cheap cheating patterns, review the "
            "flagged ones with an LLM judge (which never sees the outcome), and write "
            "cheating_check.json plus cheating_report.html. Runs and rollouts are selected "
            "with the same flags as `autofa prepare`."
        ),
        epilog=(
            "examples:\n"
            "  autofa preflight cheating\n"
            "  autofa preflight cheating --api-run RUN_UUID\n"
            "  autofa preflight cheating --input ./eval-results\n"
            "  autofa preflight cheating --model MODEL --benchmark BENCH --latest 3\n"
            "  autofa preflight cheating --api-run RUN_UUID --per-run 200 --max-reviews 20\n"
            "  autofa preflight cheating --input SELECTION_ID"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_cohort_arguments(parser, taxonomy_inputs=True)
    add_taxonomy_filter_arguments(parser)
    parser.add_argument("--api-base", dest="api_base_url", default=None,
                        help="Override the configured Data API base URL")
    parser.add_argument("--max-reviews", type=int, default=0, metavar="N",
                        help="Review a random sample of at most N flagged rollouts "
                             "(default 0: review all of them)")
    parser.add_argument("--screen-only", action="store_true",
                        help="Run the cheap patterns only; no LLM reviews")
    parser.add_argument("--judge-model",
                        help="Judge model (default: cheating_check_model setting)")
    parser.add_argument("--judge-api-key-env", metavar="NAME",
                        help="Environment variable holding the judge API key "
                             "(default: judge_api_key_env, else the hub key)")
    parser.add_argument("--output", type=Path,
                        help="Output directory (default: "
                             "$AUTOFA_HOME/preflight/cheating/<source>-<time>)")
    return parser


def _has_source(options: dict[str, Any]) -> bool:
    return any(options[k] for k in ("api_runs", "folders", "models", "benchmarks", "runs_where",
                                    "latest"))


def _select(args: argparse.Namespace, parser: argparse.ArgumentParser, api: Any) -> Any:
    """The selected rollouts, as `autofa prepare` and `autofa classify` select them."""
    from autofa.data.cohort import load_cohort, select_cohort

    options = cohort_options(args)
    input_kind, inputs = resolve_inputs(args.input)
    taxonomy_filters = bool(
        args.judge_view
        or args.group != "all"
        or args.label
        or args.evidence
        or args.list_taxonomy
    )
    if taxonomy_filters and input_kind != "taxonomy":
        parser.error("taxonomy filters require taxonomy results passed with --input")
    if input_kind in {"selection", "taxonomy"} and any((
        options["api_runs"],
        options["models"],
        options["benchmarks"],
        options["runs_where"],
        options["latest"],
        options["where"],
    )):
        parser.error(
            f"{input_kind} input cannot be combined with run or advanced WHERE filters"
        )
    if input_kind == "local":
        options["folders"] = inputs
    elif input_kind == "selection":
        return refine_cohort(
            load_cohort(inputs[0]),
            tasks=options["tasks"],
            rewards=options["rewards"],
            per_run=options["per_run"],
            sample=options["sample"],
            seed=options["seed"],
            limit=options["limit"],
            api=api,
        )
    elif input_kind == "taxonomy":
        from autofa.taxonomy.groups import classification_facets, cohort_from_classifications

        paths = [Path(value) for value in inputs]
        if args.list_taxonomy:
            facets = classification_facets(paths)
            print("Judges: " + ", ".join(facets["judges"]))
            print("Labels:")
            for code, label in sorted(facets["labels"].items()):
                print(f"  {code}: {label}")
            print("Evidence: " + ", ".join(facets["evidence"]))
            return None
        cohort = cohort_from_classifications(
            paths,
            judge=args.judge_view,
            group=args.group,
            labels=args.label,
            evidence=args.evidence,
        )
        return refine_cohort(
            cohort,
            tasks=options["tasks"],
            rewards=options["rewards"],
            per_run=options["per_run"],
            sample=options["sample"],
            seed=options["seed"],
            limit=options["limit"],
        )
    if not _has_source(options):
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            parser.error(f"pass {SOURCE_FLAGS}")
        from autofa.data.cohort import _pick_interactive_sources

        picked = _pick_interactive_sources(api)
        if picked is None:
            return None
        options["api_runs"], options["folders"] = picked
        if not (args.max_reviews or args.screen_only):
            reviews = _pick_reviews()
            if reviews is None:
                return None
            args.max_reviews = reviews
    return select_cohort(**options, api=api)


def _pick_reviews() -> int | None:
    """How many flagged rollouts the judge reviews (0: all), asked like `autofa classify` asks
    how many failures to classify. Screening stays complete, so the flagged counts cover the run."""
    from autofa.ui.menus import MenuItem, select_option

    choice = select_option("How many flagged rollouts to review",
                           [MenuItem(label, str(value)) for value, label in _REVIEW_CHOICES],
                           allow_cancel=True)
    if choice is None:
        return None
    if int(choice.value) != CUSTOM:
        return int(choice.value)
    from prompt_toolkit import prompt
    from prompt_toolkit.validation import Validator

    positive = Validator.from_callable(lambda text: text.strip().isdigit() and int(text) > 0,
                                       error_message="Enter a positive number")
    try:
        return int(prompt("Flagged rollouts to review: ", validator=positive))
    except (EOFError, KeyboardInterrupt):
        return None


@functools.cache
def _local_server() -> str:
    from autofa.config import load_config
    from autofa.local_service import dashboard_url_from_config

    return dashboard_url_from_config(load_config()).rstrip("/")


def _viewer(url: str) -> dict[str, Any]:
    """Remote links are complete. Local runs get server-relative paths (`/traces/<store>/...`),
    and the Local Library server refuses clicks from a report opened as a file, so the report
    offers their full address to copy into the address bar instead."""
    if not url.startswith("/"):
        return {"viewer_url": url}
    return {"viewer_url": _local_server() + url, "viewer_local": True}


def _load_sources(cohort: Any, api: Any) -> tuple[list[Path], list[dict[str, Any]]]:
    """Materialize the selection: parquet paths plus one viewer reference per run."""
    from autofa.data.cohort import prepare_cohort

    datasets = prepare_cohort(cohort, api=api, activity=print)
    paths = [Path(dataset["parquet_path"]) for dataset in datasets]
    return paths, datasets


def _meta(item: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, Any]:
    from autofa.data.parquet_store import _is_correct
    from autofa.data.reference_data import reference_for_parquet, rollout_url

    reference = reference_for_parquet(references, item.get("_source_parquet"))
    viewer = (rollout_url(reference, problem_id=item.get("problem_id"),
                          run_idx=item.get("run_idx")) if reference else "")
    return {
        "run_id": (reference or {}).get("run_id"), "problem_id": item.get("problem_id"),
        "run_idx": item.get("run_idx"), "benchmark": item.get("benchmark"),
        "model_id": item.get("model_id"), "passed": _is_correct(item), **_viewer(viewer),
    }


def _default_output(sources: list[dict[str, Any]]) -> Path:
    from autofa.config import autofa_home

    names = [str(s.get("run_id") or "") for s in sources]
    name = re.sub(r"[^\w.-]+", "_", "+".join(names))[:80] or "run"
    return autofa_home() / "preflight" / "cheating" / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"


def _print_plan(plan: dict[str, Any]) -> None:
    """Say how many reviews are about to start, and what they cost on earlier runs."""
    flagged = plan["flagged"] + plan["reused"]
    line = f"{flagged} flagged rollout{'s' * (flagged != 1)} → reviewing {plan['reviewing']}"
    per = plan["tokens_per_review"]
    if plan["reviewing"] and per:
        total = plan["reviewing"] * per / 1e6
        line += (f" (~{total:,.{0 if total >= 10 else 1}f}M judge tokens at the median "
                 f"{per / 1e6:.2f}M per earlier review)")
    if plan["reused"]:
        line += f", reusing {plan['reused']} earlier verdict{'s' * (plan['reused'] != 1)}"
    print(line)
    if plan["reviewing"] > 100 and plan["reviewing"] == plan["flagged"]:
        print("Use --max-reviews N to review a random sample instead.")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "under a minute"
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


def _review_status(done: int, total: int, tokens: int, elapsed: float, workers: int) -> str:
    """Reviews done, judge tokens spent, and the time left at the pace so far. Reviews run
    `workers` at a time, so the pace is known once the first of them have finished."""
    spent = f"{tokens / 1e6:.1f}M" if tokens >= 100_000 else f"{tokens:,}"
    if done >= total:
        left = f"took {_duration(elapsed)}"
    elif done < min(workers, total):
        left = "estimating time left"
    else:
        left = f"{_duration(elapsed / done * (total - done))} left"
    return f"{done}/{total} · {spent} judge tokens · {left}"


class _ReviewProgress:
    """How far the LLM reviews are: a live bar in a terminal, else a line every tenth of them."""

    def __init__(self, workers: int, *, live: bool | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.workers, self.clock = workers, clock
        self.live = sys.stdout.isatty() if live is None else live
        self.total = self.done = self.tokens = 0
        self.start = 0.0
        self.bar: Any = None

    def plan(self, plan: dict[str, Any]) -> None:
        _print_plan(plan)
        self.total, self.start = plan["reviewing"], self.clock()
        if self.total and self.live:
            self._start_bar()

    def review(self, result: dict[str, Any]) -> None:
        self.done += 1
        self.tokens += int(result.get("tokens") or 0)
        if self.bar is not None:
            self.bar.update(self.task, completed=self.done, status=self._status())
        elif self.done < self.total and self.done % max(1, self.total // 10) == 0:
            print(f"Reviewed {self._status()}")

    def __enter__(self) -> _ReviewProgress:
        return self

    def __exit__(self, *exc: object) -> None:
        # Best effort: the output may be gone, and the report is only written after this.
        with suppress(OSError, ValueError):
            if self.bar is not None:
                self.bar.stop()
            # Only a display that saw every review has the right totals: one switched off early
            # would print stale counts.
            if self.done and self.done == self.total:
                print(f"Reviewed {self._status()}")

    def _status(self) -> str:
        return _review_status(self.done, self.total, self.tokens, self.clock() - self.start,
                              self.workers)

    def _start_bar(self) -> None:
        from rich.progress import BarColumn, Progress, TextColumn

        from autofa.ui.theme import GREEN

        self.bar = Progress(TextColumn("Reviewing"), BarColumn(complete_style=GREEN),
                            TextColumn("{task.fields[status]}"), transient=True)
        self.task = self.bar.add_task("", total=self.total, status=self._status())
        self.bar.start()


def _count(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def _verdict(s: dict[str, Any]) -> tuple[str, str]:
    """What the check found, in one line: cheating (bad), no cheating (good), or not settled
    (open). "No cheating found" is said only when every flagged rollout was settled; otherwise
    the line says how many flagged rollouts stay open, and why. The counts line gives the rest."""
    flagged, skipped = s["escalated"], s["not_reviewed"]
    failed, unsure = s["unjudged"], (s.get("statuses") or {}).get("insufficient_evidence", 0)
    left = failed + unsure + skipped
    details = ", ".join(part for part in (
        failed and _count(failed, "review") + " failed",
        unsure and f"{unsure} with insufficient evidence",
        skipped and f"{skipped} not reviewed") if part)
    stay = f"{_count(left, 'flagged rollout')} {'stays' if left == 1 else 'stay'} open ({details})"
    if s["cheated"]:
        # With a review sample, cheating is counted among the reviewed rollouts only.
        among = (f"the {_count(flagged - skipped, 'reviewed rollout')}" if skipped
                 else _count(s["rollouts"], "rollout"))
        passes = s["contaminated_passes"]
        line = f"Cheating found in {s['cheated']} of {among}"
        line += f" ({_count(passes, 'contaminated pass', 'contaminated passes')})" if passes else ""
        return line + (f"; {stay}." if left else "."), "bad"
    if left:
        return f"Not settled: {stay}.", "open"
    return "No cheating found.", "good"


def _print_verdict(summary: dict[str, Any]) -> None:
    text, tone = _verdict(summary)
    if not sys.stdout.isatty():
        print(text)
        return
    from rich.console import Console
    from rich.text import Text

    from autofa.ui.theme import GREEN, WARN

    style = {"bad": "bold red", "good": f"bold {GREEN}", "open": f"bold {WARN}"}[tone]
    Console().print(Text(text, style=style))


def _open_report(page: Path) -> None:
    """Open the report in a browser, for the standalone command and analysis runs alike. On Linux
    without a display, Python may pick a text browser that holds the terminal until it is quit,
    so there only the printed link is left. A browser that cannot open only costs a warning."""
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    url = page.as_uri()
    try:
        opened = webbrowser.open(url)
    except (webbrowser.Error, ValueError, OSError):  # e.g. a malformed BROWSER setting
        opened = False
    if not opened:
        print(f"Could not open the report automatically: {url}")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from autofa.config import load_config
    from autofa.data.cohort import save_cohort
    from autofa.data.remote_api import DataApiClient

    if args.max_reviews < 0:
        raise ValueError("--max-reviews must be >= 0")
    if args.seed and args.sample != "random":
        raise ValueError("--seed requires --sample random")
    if args.sample == "random" and not args.per_run:
        raise ValueError("--sample random requires --per-run")
    cfg = load_config()
    if args.judge_model:
        cfg.cheating_check_model = args.judge_model
    if args.judge_api_key_env:
        cfg.judge_api_key_env = args.judge_api_key_env
    api = DataApiClient(args.api_base_url or cfg.data_api_base_url)
    cohort = _select(args, parser, api)
    if cohort is None:
        return 0
    for source in cohort.sources:
        print(f"{source['kind']} run {source['run_id']}: "
              f"{cohort.count(source['run_id'])} rollout(s) selected")
    if cohort.empty:
        raise ValueError("No rollouts matched")
    save_cohort(cohort)
    print(f"Selection {cohort.id}: {cohort.count()} rollout(s)")
    paths, sources = _load_sources(cohort, api)
    return run_dataset(
        Dataset(parquet_paths=paths), cfg, args.output or _default_output(sources),
        sources=sources,
        max_reviews=0 if args.screen_only else args.max_reviews or None,
    ).status


def run_dataset(dataset: Dataset, cfg: Config, output: Path, *,
                sources: list[dict[str, Any]], max_reviews: int | None = None) -> PreflightResult:
    """Shared detector entrypoint for standalone commands and configured analysis runs."""
    from autofa.data.reference_data import inspect_url
    from autofa.preflight.cheating.pipeline import judge_items, summarize
    from autofa.preflight.cheating.report import rollout_rows, write_report

    sources = [{**source, **_viewer(inspect_url(source))} for source in sources]
    references = [s for s in sources if s.get("run_id")]
    items, meta = [], {}
    for batch in dataset.iter_batches():
        for item in batch:
            items.append(item)
            meta[str(item["_item_id"])] = _meta(item, references)
    out_dir = output.expanduser().resolve()
    passed = {iid for iid, m in meta.items() if m["passed"]}
    print(f"Checking {len(items)} rollouts ({len(passed)} passed) "
          f"from {len(dataset.parquet_paths)} source(s)…")
    with _ReviewProgress(max(1, cfg.judge_workers)) as progress:
        results = judge_items(items, cfg, out_dir, max_reviews=max_reviews,
                              on_plan=progress.plan, on_review=progress.review)
    summary = summarize(results, cfg, passed)
    data, page = write_report(out_dir, summary=summary, rows=rollout_rows(results, meta),
                              sources=sources)
    print(f"{summary['rollouts']} rollouts ({summary['passed']} passed) · "
          f"{summary['escalated']} flagged · {summary['cheated']} cheated · "
          f"{summary['contaminated']} could have caused a pass "
          f"({summary['contaminated_passes']} of them passed) · "
          f"{summary['unjudged']} review failed · {summary['not_reviewed']} not reviewed")
    _print_verdict(summary)
    print(f"Report: {page.as_uri()}")
    _open_report(page)
    if summary["unjudged"]:
        print("warning: some reviews failed; see the 'Needs attention' section", file=sys.stderr)
    reviewed = sum(1 for r in results.values() if r.get("escalated") and not r.get("cached")
                   and r["status"] != "not_reviewed")
    # Every review failing usually means no access to the judge model, not an unlucky run.
    return PreflightResult(page, data, 1 if reviewed and summary["unjudged"] == reviewed else 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, parser)
    except (OSError, DataApiError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
