# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``autofa classify`` command line."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shlex
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from contextvars import copy_context
from copy import copy
from pathlib import Path
from typing import Any

from autofa.data.cohort import (
    _HEALTHY_SCORED,
    Cohort,
    add_cohort_arguments,
    add_taxonomy_filter_arguments,
    cohort_options,
    cohorts_root,
    load_cohort,
    matches_reward_filters,
    parse_reward_filters,
    prepare_data,
    query_remote_run,
    refine_cohort,
    remote_run_uuid,
    resolve_inputs,
    save_cohort,
    select_cohort,
)
from autofa.data.outcomes import is_healthy_scored
from autofa.data.parquet_store import Dataset
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.source_errors import SourceDataError
from autofa.diagnostics import get_diagnostics

from . import report
from .classifier import PROTOCOL_VERSION, TaxonomyClassifier
from .manifest import write_manifest
from .outcome_evidence import OutcomeSource
from .pipeline import (
    EVIDENCE_CACHE_VERSION,
    classify_prepared,
    prepare_rollout,
)
from .providers import (
    OpenAIProvider,
    ProviderError,
    effective_reasoning_effort,
    image_input_is_unsupported,
    provider_configuration_sha256,
)
from .record import (
    append_classification,
    classification_record,
    classification_sha256,
    has_usable_first_pass,
    read_classifications,
)
from .schema import DEFAULT_TAXONOMY_PATH, TaxonomyError, load_taxonomy
from .sources import LocalOutcomeSource, RemoteOutcomeSource
from .trace_digest import DEFAULT_JUDGEMENT_CHARACTER_BUDGET

_TAXONOMY_HELP = "Custom taxonomy JSON (default: built-in)"
_ERRORS = (OSError, DataApiError, SourceDataError, ProviderError, ValueError, RuntimeError)
DEFAULT_PER_RUN = 10
JUDGE_TIMEOUT_SECONDS = 300.0
JUDGE_PROBE_TIMEOUT_SECONDS = 30.0
DATA_API_TIMEOUT_SECONDS = 120.0
DEFAULT_WORKERS = 4
# Judge errors that every remaining rollout would repeat.
_FATAL_JUDGE_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}
_COUNT_CHOICES = (
    (DEFAULT_PER_RUN, f"{DEFAULT_PER_RUN} random failures per run"),
    (50, "50 random failures per run"),
    (0, "All failures"),
)
log = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    incoming = list(sys.argv[1:] if argv is None else argv)
    if incoming and incoming[0] == "report":
        return report.main(incoming[1:])
    return classify_main(incoming)


def _configure_logging(logs: ExitStack, destination: Path) -> None:
    logs.enter_context(get_diagnostics().configured(destination.parent, console=True))
    log.info("taxonomy classification output: %s", destination)


def _status(
    message: str,
    *,
    level: int = logging.INFO,
    flush: bool = False,
) -> None:
    root = logging.getLogger("autofa")
    if not any(getattr(handler, "_autofa_console", False) for handler in root.handlers):
        print(message, file=sys.stderr if level >= logging.WARNING else sys.stdout, flush=flush)
    log.log(level, message)


def classify_main(  # noqa: C901
    argv: Sequence[str] | None = None,
    *,
    config: Any | None = None,
    workflow_config: Mapping[str, Any] | None = None,
    on_selection: Callable[[Cohort], None] | None = None,
) -> int:
    parser = build_classify_parser()
    incoming = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(incoming)
    args._workflow_config = dict(workflow_config) if workflow_config else None
    args.sample_explicit = any(
        value == "--sample" or value.startswith("--sample=") for value in incoming
    )
    logs = ExitStack()
    try:
        _validate_shard_args(args)
        if args.prefetch_only and args.evidence_cache is None:
            raise ValueError("--prefetch-only requires --cache")
        from autofa.config import load_config

        cfg = config or load_config()
        if (
            not incoming
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and not _select_interactive_workflow(args, cfg)
        ):
            return 0
        api = DataApiClient(
            args.api_base_url or cfg.data_api_base_url, timeout=DATA_API_TIMEOUT_SECONDS
        )
        cohort = _select(
            args,
            parser,
            api,
            api_name=cfg.data_api_name,
            taxonomy_second_pass=cfg.taxonomy_second_pass,
        )
        if cohort is None:
            if getattr(args, "_second_pass_parents", None) is None:
                return 0
            if config is not None:
                return _classify_results_from_cache(args, logs, config=cfg)
            return _classify_results_from_cache(args, logs)
        _validate_classification_cohort(cohort, args.reward)
        if on_selection is not None:
            on_selection(cohort)
        if not args.reward and not args.dry_run:
            _warn_partial_rewards(cohort, api)
        destination = _classification_output(args.output, args.run, cohort)
        if not args.dry_run:
            _configure_logging(logs, destination)
        args._defer_judge_preflight = bool(
            args.resume and destination.is_file() or args.resume_from
        )
        classifier = None
        if not args.prefetch_only and not args.dry_run:
            classifier = (
                _classifier(args, config=cfg) if config is not None else _classifier(args)
            )
        if classifier is not None:
            _validate_output_profile(
                destination,
                classifier,
                resume=args.resume,
                parent_configuration=_saved_parent_configuration(args),
            )
            if args.resume:
                args._resume_review_parents = _review_failed_parents(
                    destination,
                    _classification_profile(classifier),
                )
        work_cohort = _classification_work_cohort(
            args,
            cohort,
            destination,
            classifier,
        )
        for source in work_cohort.sources:
            count = work_cohort.count(source["run_id"])
            _status(f"{source['kind']} run {source['run_id']}: {count} rollout(s) selected")
        if work_cohort.empty:
            if not args.dry_run:
                save_cohort(cohort)
                if not args.prefetch_only and destination.is_file():
                    _status(f"wrote 0 to {destination}: 0 succeeded, 0 failed")
                    return _write_outputs(
                        args,
                        destination,
                        0,
                        cohort,
                        classifier,
                    )
            _status("No rollouts matched")
            return 0
        if args.dry_run:
            return 0
        if classifier is not None and args._defer_judge_preflight:
            _preflight_classifier(
                classifier,
                methods=_pending_preflight_methods(args, work_cohort, classifier),
            )
        save_cohort(cohort)
        _status(
            f"Selection {cohort.id}: {work_cohort.count()} rollout(s) pending",
            flush=True,
        )
        try:
            status = 0
            if args.evidence_cache is not None:
                work_cohort, status = _classify_cached_rows(
                    args, cohort, work_cohort, classifier, destination
                )
            if not work_cohort.empty:
                prepared = prepare_data(work_cohort, api=api, activity=_activity)
                if work_cohort is not cohort:
                    work_sources = {
                        source["run_id"]: source for source in work_cohort.sources
                    }
                    for source in cohort.sources:
                        source.update(work_sources.get(source["run_id"], {}))
                    cohort.datasets = list(work_cohort.datasets)
                    save_cohort(cohort)
                status = max(
                    status,
                    _classify_dataset(
                        args,
                        cohort,
                        prepared.reference_data,
                        api,
                        classifier,
                        "chat",
                        destination,
                    ),
                )
        except RuntimeError:
            if destination.is_file() and classifier is not None:
                _write_manifest(args, destination, cohort, classifier)
            raise
        return (
            status
            if args.prefetch_only
            else _write_outputs(args, destination, status, cohort, classifier)
        )
    except TaxonomyError as exc:
        _print_taxonomy_error(exc, args.taxonomy)
        return 2
    except _ERRORS as exc:
        _status(f"error: {exc}", level=logging.ERROR)
        return 2
    finally:
        logs.close()


def run_taxonomy_workflow(workflow: Any) -> int:
    """Run selection and publish each configured classification workflow."""
    from autofa.config import load_config

    base_config = load_config()
    if workflow.judge_base_url is not None:
        base_config.judge_base_url = workflow.judge_base_url
    if workflow.judge_api_key_env is not None:
        base_config.judge_api_key_env = workflow.judge_api_key_env

    source = workflow.source
    selection = workflow.selection
    command = []
    if source.local_folder is not None:
        command.extend(["--input", str(source.local_folder)])
    else:
        for run_id in source.api_runs:
            command.extend(["--api-run", run_id])
        if source.api_base_url:
            command.extend(["--api-base", source.api_base_url])
    command.extend([
        "--per-run", str(selection.per_run),
        "--sample", selection.sample,
        "--seed", str(selection.seed),
        "--cache", str(selection.cache),
        "--prefetch-only",
    ])
    selected: list[Cohort] = []
    status = classify_main(command, config=base_config, on_selection=selected.append)
    if status:
        return status
    if len(selected) != 1:
        raise RuntimeError("prefetch did not return its rollout selection")
    selected_id = selected[0].id

    final_status = 0
    for configured in workflow.runs:
        snapshot = _workflow_config_snapshot(workflow, configured)
        outputs = []
        for judge in configured.judges:
            _status(
                f"Running {configured.first_pass}/{configured.second_pass} judge "
                f"{judge.name}…",
                flush=True,
            )
            config = copy(base_config)
            if judge.harness is not None:
                config.behavior_judge_harness = judge.harness
            command = [
                "--input", selected_id,
                "--cache", str(selection.cache),
                "--first-pass", configured.first_pass,
                "--second-pass", configured.second_pass,
                "--workers", str(judge.workers),
                "--output", str(judge.output),
            ]
            if judge.model is not None:
                command.extend(["--judge-model", judge.model])
            for value in judge.params:
                command.extend(["--judge-param", value])
            if judge.output.is_file():
                command.append("--resume")
            status = classify_main(
                command,
                config=config,
                workflow_config=snapshot,
            )
            if status:
                return status
            outputs.append(str(judge.output))

        report_args = [*outputs, "--output", str(configured.report.output)]
        if configured.report.publish:
            report_args.extend(["--publish", configured.report.publish])
        status = report.main(report_args)
        if status:
            if not configured.report.output.is_file():
                return status
            final_status = status
    return final_status


def _workflow_config_snapshot(workflow: Any, configured: Any) -> dict[str, Any]:
    """Return the reproducible, credential-free config shown in report provenance."""
    selection = workflow.selection

    def display_path(path: Path) -> str:
        value = str(path)
        home = str(Path.home())
        return f"~{value[len(home):]}" if value.startswith(home) else value

    source = workflow.source
    if source.evalrdl_url is not None:
        source_snapshot = {"evalrdl_url": source.evalrdl_url}
    elif source.local_folder is not None:
        source_snapshot = {"local_folder": display_path(source.local_folder)}
    else:
        source_snapshot = {"api_runs": list(source.api_runs)}
    return {
        "version": 2,
        "workflow": "classification",
        "source": source_snapshot,
        "selection": {
            "per_run": selection.per_run,
            "sample": selection.sample,
            "seed": selection.seed,
            "cache": display_path(selection.cache),
        },
        "taxonomy": {
            **(
                {"judge_base_url": workflow.judge_base_url}
                if workflow.judge_base_url is not None
                else {}
            ),
            **(
                {"judge_api_key_env": workflow.judge_api_key_env}
                if workflow.judge_api_key_env is not None
                else {}
            ),
            "runs": [{
                "first_pass": configured.first_pass,
                "second_pass": configured.second_pass,
                "judges": [
                    {
                        "name": judge.name,
                        **({"model": judge.model} if judge.model is not None else {}),
                        "harness": judge.harness,
                        "params": list(judge.params),
                        "workers": judge.workers,
                        "output": display_path(judge.output),
                    }
                    for judge in configured.judges
                ],
                "report": {
                    "output": display_path(configured.report.output),
                    "publish": configured.report.publish,
                },
            }],
        },
    }


def _has_selection(args: argparse.Namespace) -> bool:
    return any(
        (
            args.api_run,
            args.model,
            args.benchmark,
            args.runs_where,
            args.latest,
            args.task,
            args.where,
            args.per_run,
        )
    )


def _select(  # noqa: C901
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    api: DataApiClient,
    *, api_name: str = "EvalRDL remote server",
    taxonomy_second_pass: str = "llm",
) -> Cohort | None:
    """Select failed rollouts, sampling by default."""
    input_kind, inputs = resolve_inputs(args.input)
    if input_kind == "taxonomy":
        if getattr(args, "first_pass", None) is not None:
            parser.error("--first-pass applies only to fresh rollout evidence")
        if args.list_taxonomy:
            from .groups import classification_facets

            facets = classification_facets([Path(value) for value in inputs])
            print("Judges: " + ", ".join(facets["judges"]))
            print("Labels:")
            for code, label in sorted(facets["labels"].items()):
                print(f"  {code}: {label}")
            print("Evidence: " + ", ".join(facets["evidence"]))
            return None
        if any((
            args.api_run,
            args.model,
            args.benchmark,
            args.runs_where,
            args.latest,
            args.where,
        )):
            parser.error(
                "taxonomy results cannot be combined with run or advanced WHERE filters"
            )
        method = args.second_pass or taxonomy_second_pass
        if method == "off":
            parser.error("taxonomy results require --second-pass llm or agentic")
        from .groups import (
            classification_rollout_identities,
            cohort_from_classifications,
            resolve_classification_paths,
            saved_selection_id,
        )

        paths = resolve_classification_paths([Path(value) for value in inputs])
        selection_id = saved_selection_id(paths)
        if args.evidence_cache is not None and (
            selection_id is None
            or not (cohorts_root() / selection_id / "cohort.json").is_file()
        ):
            if (
                args.prefetch_only
                or _has_selection(args)
                or args.all
                or args.seed
                or args.sample != "first"
            ):
                parser.error(
                    "the saved selection behind these results is not on this machine, "
                    "so they can only be reviewed from --cache; drop "
                    "--prefetch-only and sampling options"
                )
            selected = classification_rollout_identities(
                paths,
                judge=args.judge_view,
                group=args.group,
                labels=args.label,
                evidence=args.evidence,
                include_review_failed=True,
            )
            args._second_pass_parents = _taxonomy_parent_records(
                paths,
                selected,
                judge=args.judge_view,
            )
            args._saved_first_pass_method = _parent_first_pass_method(
                args._second_pass_parents
            )
            if not args.output and len(paths) == 1:
                source = paths[0].expanduser().resolve()
                args.output = str(
                    source.with_name(f"{source.stem}.second-pass-{method}.jsonl")
                )
            return None
        cohort = cohort_from_classifications(
            paths,
            judge=args.judge_view,
            group=args.group,
            labels=args.label,
            evidence=args.evidence,
            include_review_failed=True,
        )
        cohort = refine_cohort(
            cohort,
            tasks=args.task,
            rewards=args.reward or ["<=0"],
            per_run=args.per_run,
            sample=args.sample,
            seed=args.seed,
            limit=0,
            api=api,
        )
        args._second_pass_parents = _taxonomy_parent_records(
            paths,
            {
                (str(row["run_id"]), str(row["task_id"]), int(row["repeat"]))
                for row in cohort.rows
            },
            judge=args.judge_view,
        )
        args._saved_first_pass_method = _parent_first_pass_method(
            args._second_pass_parents
        )
        if not args.output and len(paths) == 1:
            source = paths[0].expanduser().resolve()
            args.output = str(
                source.with_name(f"{source.stem}.second-pass-{method}.jsonl")
            )
        return cohort
    if (
        args.judge_view
        or args.group != "all"
        or args.label
        or args.evidence
        or args.list_taxonomy
    ):
        parser.error("taxonomy filters require taxonomy results passed with --input")
    if input_kind == "selection" and any((
        args.api_run,
        args.model,
        args.benchmark,
        args.runs_where,
        args.latest,
        args.where,
    )):
        parser.error("saved selections cannot be combined with run or advanced WHERE filters")
    if args.all and args.per_run:
        parser.error("--all cannot be combined with --per-run")
    options = cohort_options(args)
    options["healthy_scored"] = True
    options["failures"] = not bool(args.reward)
    if input_kind == "local":
        options["folders"] = inputs
    has_source = any(
        (
            inputs,
            options["api_runs"],
            options["folders"],
            options["models"],
            options["benchmarks"],
            options["runs_where"],
            options["latest"],
        )
    )
    interactive = not has_source
    if interactive:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            parser.error(
                "pass --api-run, --input, --model, --benchmark, --runs-where "
                "or --latest"
            )
        from autofa.data.cohort import _pick_interactive_sources

        picked = _pick_interactive_sources(api, api_name=api_name)
        if picked is None:
            return None
        options["api_runs"], options["folders"] = picked
    if input_kind == "selection":
        if args.sample == "random" and not args.per_run:
            parser.error("--sample random requires --per-run")
        if args.seed and args.sample != "random":
            parser.error("--seed requires random sampling")
        return refine_cohort(
            load_cohort(inputs[0]),
            tasks=options["tasks"],
            rewards=args.reward or ["<=0"],
            per_run=args.per_run,
            sample=args.sample,
            seed=args.seed,
            api=api,
        )
    if not args.all and not args.per_run and not args.task:
        per_run = _pick_count() if interactive else DEFAULT_PER_RUN
        if per_run is None:
            return None
        if per_run and not interactive:
            _status(
                f"Classifying up to {per_run} random rollouts per run; "
                "pass --all for every match or --per-run N.",
                flush=True,
            )
        options["per_run"] = per_run
        options["sample"] = (
            args.sample if args.sample_explicit else "random" if per_run else args.sample
        )
    elif args.per_run and not args.sample_explicit:
        options["sample"] = "random"
    if options["sample"] == "random" and not options["per_run"]:
        parser.error("--sample random requires --per-run")
    if args.seed and options["sample"] != "random":
        parser.error("--seed requires random sampling")
    # Classification applies --limit after completed resume records are excluded.
    # Keeping the cohort unbounded also gives repeated runs a stable output location.
    options["limit"] = 0
    if args.dry_run and options["folders"]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="autofa-classify-") as root:
            return select_cohort(**options, api=api, local_store_root=Path(root))
    return select_cohort(**options, api=api)


def _taxonomy_parent_records(
    paths: list[Path],
    selected: set[tuple[str, str, int]],
    *,
    judge: str | None,
) -> dict[str, dict[str, Any]]:
    """Return one latest source classification for each selected rollout."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        for record in read_classifications(path):
            if has_usable_first_pass(record):
                rows.append({**record, "_results_path": str(path.resolve())})
    views = report.classification_judge_views(
        rows,
        include_review_failed=True,
    )
    if judge is None:
        if len(views) != 1:
            available = ", ".join(views) or "none"
            raise ValueError(
                "second pass over multi-judge results requires --judge-view; "
                f"available judges: {available}"
            )
        configuration = next(iter(views.values()))
    else:
        try:
            configuration = views[judge]
        except KeyError:
            raise ValueError(
                f"judge {judge!r} was not found; available judges: {', '.join(views)}"
            ) from None
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if report._judge_configuration(row) != configuration:
            continue
        source = row.get("source") or {}
        try:
            identity = (
                str(source["run_id"]),
                str(source["task_id"]),
                int(source["repeat"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        latest[identity] = row
    missing = selected - set(latest)
    if missing:
        run_id, task_id, repeat = sorted(missing)[0]
        raise ValueError(
            f"taxonomy results have no parent classification for "
            f"{run_id}:{task_id}:{repeat}"
        )
    return {
        f"{run_id}:{task_id}:{repeat}": latest[(run_id, task_id, repeat)]
        for run_id, task_id, repeat in selected
    }


def _parent_first_pass_method(parents: dict[str, dict[str, Any]]) -> str:
    methods = {_record_first_pass_method(parent) for parent in parents.values()}
    if len(methods) != 1:
        raise ValueError("selected taxonomy results mix llm and agentic first passes")
    return methods.pop()


def _activity(message: str) -> None:
    _status(message, flush=True)


def _select_interactive_workflow(args: argparse.Namespace, cfg: Any) -> bool:
    from autofa.ui.taxonomy_settings import select_taxonomy_workflow

    workflow = select_taxonomy_workflow(
        cfg.taxonomy_first_pass,
        cfg.taxonomy_second_pass,
    )
    if workflow is None:
        return False
    args.first_pass, args.second_pass = workflow
    return True


def _validate_classification_cohort(cohort: Cohort, rewards: Sequence[str]) -> None:
    """Reject selections containing unhealthy, unscored, or out-of-range rows."""
    if any(source.get("selection") == "all" for source in cohort.sources):
        raise ValueError(
            "saved selection contains the entire run without rollout identities; "
            "select the source run again"
        )
    filters = parse_reward_filters(rewards or ["<=0"])
    for row in cohort.rows:
        healthy = is_healthy_scored(row)
        if not healthy or not matches_reward_filters(row.get("reward"), filters):
            rollout_id = (
                f"{row.get('run_id')}:{row.get('task_id')}:{int(row.get('repeat') or 0)}"
            )
            raise ValueError(
                f"selection contains a rollout outside the classification reward filter: "
                f"{rollout_id}"
            )


def _classification_work_cohort(
    args: argparse.Namespace,
    cohort: Cohort,
    destination: Path,
    classifier: TaxonomyClassifier | None,
) -> Cohort:
    """Return only pending, sharded rows that this invocation can process."""
    done = _resume_rollout_ids(args, destination, classifier)
    rows = [
        dict(row)
        for row in cohort.rows
        if (
            (
                rollout_id := (
                    f"{row['run_id']}:{row['task_id']}:{int(row['repeat'])}"
                )
            )
            not in done
            and _rollout_in_shard(rollout_id, args.shard_count, args.shard_index)
        )
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    return _cohort_rows(cohort, rows, limit=args.limit)


def _cohort_rows(cohort: Cohort, rows: list[dict[str, Any]], *, limit: int) -> Cohort:
    """Return the selection narrowed to ``rows``, or itself when nothing was removed."""
    if rows == cohort.rows:
        return cohort
    selected_runs = {str(row["run_id"]) for row in rows}
    return Cohort(
        runs_where=cohort.runs_where,
        where=cohort.where,
        sources=[
            dict(source)
            for source in cohort.sources
            if str(source["run_id"]) in selected_runs
        ],
        rows=rows,
        latest=cohort.latest,
        per_run=cohort.per_run,
        sample=cohort.sample,
        seed=cohort.seed,
        limit=limit,
        created_at=cohort.created_at,
        parent_selection=cohort.id,
        taxonomy_filter=dict(cohort.taxonomy_filter),
    )


def _warn_partial_rewards(cohort: Cohort, api: DataApiClient) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from autofa.data.local_trace_store import resolve_store

    partial_runs = []
    for source in cohort.sources:
        run_id = str(source["run_id"])
        if source["kind"] == "remote":
            client = (
                api
                if api.base_url == source["base_url"]
                else DataApiClient(source["base_url"], timeout=DATA_API_TIMEOUT_SECONDS)
            )
            result, _ = query_remote_run(
                client, run_id, remote_run_uuid(client, source), "SELECT task_id FROM di.rollouts",
                [_HEALTHY_SCORED, "reward > 0 AND reward < 1"], "LIMIT 1",
            )
            if result.get("rows"):
                partial_runs.append(run_id)
            continue
        try:
            store = resolve_store(source["store_id"])
        except ValueError:
            continue
        table = pq.read_table(
            store / "tables" / "rollouts.parquet",
            columns=["reward", "status", "ingest_status", "error", "is_scored"],
        )
        mask = pc.equal(table["status"], "ok")
        for condition in (
            pc.equal(table["ingest_status"], "ok"),
            pc.is_null(table["error"]),
            pc.equal(table["is_scored"], True),
            pc.greater(table["reward"], 0),
            pc.less(table["reward"], 1),
        ):
            mask = pc.and_kleene(mask, condition)
        if int(pc.sum(mask).as_py() or 0):
            partial_runs.append(run_id)
    if partial_runs:
        _status(
            "warning: partial rewards exist in "
            + ", ".join(partial_runs)
            + "; the default selects only reward <= 0. "
            "Pass --reward '<1' or another threshold to include them.",
            level=logging.WARNING,
        )


def _pick_count() -> int | None:
    from autofa.ui.menus import MenuItem, select_option

    choice = select_option(
        "How many failures to classify",
        [MenuItem(label, value) for value, label in _COUNT_CHOICES],
        allow_cancel=True,
    )
    return None if choice is None else int(choice.value)


def build_classify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autofa classify",
        description="Classify evaluation rollouts and append JSONL records.",
        epilog=(
            "Classifies scored failures (reward <= 0). Without --all or --per-run, a\n"
            f"random sample of {DEFAULT_PER_RUN} failures per run is classified.\n\n"
            "examples:\n"
            "  autofa classify\n"
            "  autofa classify --input ./eval-results\n"
            "  autofa classify --api-run RUN_UUID --all --resume\n"
            "  autofa classify --input SELECTION_ID\n"
            "  autofa classify report RESULTS.jsonl [MORE.jsonl | FOLDER ...]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_cohort_arguments(
        parser,
        taxonomy_inputs=True,
        limit_help="Classify at most N pending rollouts after resume and shard filtering",
    )
    add_taxonomy_filter_arguments(parser)
    parser.add_argument("--all", action="store_true", help="Classify every matching rollout")
    parser.add_argument(
        "--api-base",
        dest="api_base_url",
        default=None,
        help="Override the configured EvalRDL server base URL",
    )
    _add_judge_arguments(parser)
    parser.add_argument(
        "--resume", action="store_true", help="Skip rollouts whose latest record is ok"
    )
    parser.add_argument(
        "--cache",
        dest="evidence_cache",
        type=Path,
        metavar="DIR",
        help="Evidence folder to reuse and fill",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Fill --cache without judging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selection without downloading or judging",
    )
    parser.add_argument("--run", help="Save results under this run directory")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Results file (default: in the selection folder)",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--workers",
        type=int,
        metavar="N",
        default=DEFAULT_WORKERS,
        help=f"Parallel judge calls (default {DEFAULT_WORKERS})",
    )
    advanced.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Also skip rollouts classified in this file or folder",
    )
    advanced.add_argument(
        "--shard-count",
        type=int,
        metavar="N",
        default=1,
        help="Split the rollouts into N parts",
    )
    advanced.add_argument(
        "--shard-index",
        type=int,
        metavar="I",
        default=0,
        help="Part to run, from 0 (default 0)",
    )
    advanced.add_argument(
        "--window-max-chars",
        dest="max_characters",
        metavar="N",
        type=int,
        default=DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
        help=f"Trace chars per judge call (default {DEFAULT_JUDGEMENT_CHARACTER_BUDGET})",
    )
    advanced.add_argument(
        "--agentic-max-turns",
        type=_agentic_max_turns,
        default=argparse.SUPPRESS,
        metavar="N|none",
        help=(
            "Investigation-tool calls per agentic pass; none removes the limit "
            "(default: Settings)"
        ),
    )
    return parser


def _add_judge_arguments(parser: argparse.ArgumentParser) -> None:
    parser = parser.add_argument_group("judge")
    parser.add_argument("--taxonomy", metavar="FILE", help=_TAXONOMY_HELP)
    parser.add_argument(
        "--first-pass",
        choices=("llm", "agentic"),
        help="llm, or agentic with Claude Code or Codex (default: Settings)",
    )
    parser.add_argument(
        "--second-pass",
        choices=("off", "llm", "agentic"),
        help="Review the first pass: off, llm or agentic (default: Settings)",
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        help="Judge model (default: Settings)",
    )
    parser.add_argument(
        "--judge-param",
        dest="judge_params",
        type=_judge_param,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra llm request parameter, repeatable, e.g. reasoning_effort=high",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=JUDGE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Seconds per llm judge call (default {JUDGE_TIMEOUT_SECONDS:g})",
    )


def _judge_param(text: str) -> tuple[str, Any]:
    import tomllib

    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("use KEY=VALUE")
    try:
        parsed = tomllib.loads(f"value = {value}")["value"]
    except tomllib.TOMLDecodeError:
        parsed = value
    return key.strip(), parsed


def _agentic_max_turns(text: str) -> int | None:
    if text.strip().lower() == "none":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a positive integer, or none") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("use a positive integer, or none")
    return value


def _classifier(
    args: argparse.Namespace, *, config: Any | None = None,
) -> TaxonomyClassifier:
    from autofa.config import (
        load_config,
        validate_taxonomy_agentic_max_turns,
    )

    cfg = config or load_config()
    taxonomy = load_taxonomy(args.taxonomy)
    first_pass = getattr(args, "first_pass", None) or cfg.taxonomy_first_pass
    saved_first_pass = getattr(args, "_saved_first_pass_method", None)
    requested_second_pass = getattr(args, "second_pass", None)
    second_pass = requested_second_pass or cfg.taxonomy_second_pass
    uses_llm = (saved_first_pass is None and first_pass == "llm") or second_pass == "llm"
    uses_agentic = (
        (saved_first_pass is None and first_pass == "agentic") or second_pass == "agentic"
    )
    if args.judge_params and not uses_llm:
        raise ValueError("--judge-param applies to llm passes only")
    if args.judge_model and not uses_llm:
        raise ValueError("--judge-model applies to llm passes only")
    provider = None
    if uses_llm:
        provider = _provider(args, config=cfg) if config is not None else _provider(args)
        if not getattr(args, "_defer_judge_preflight", False):
            _check_judge(provider)
    agentic_provider = None
    if uses_agentic:
        from .agentic_review import AgenticClassificationProvider

        turns = (
            args.agentic_max_turns
            if hasattr(args, "agentic_max_turns")
            else cfg.taxonomy_agentic_max_turns
        )
        agentic_kwargs = {
            "max_turns": (
                None if turns is None else validate_taxonomy_agentic_max_turns(turns)
            ),
        }
        if getattr(args, "_defer_judge_preflight", False):
            agentic_provider = AgenticClassificationProvider.from_config(
                cfg,
                **agentic_kwargs,
                preflight=False,
            )
        else:
            agentic_provider = AgenticClassificationProvider.from_config(
                cfg,
                **agentic_kwargs,
            )
    return TaxonomyClassifier(
        taxonomy,
        provider,
        first_pass=saved_first_pass or first_pass,
        second_pass=second_pass,
        agentic_provider=agentic_provider,
        saved_first_pass=saved_first_pass,
    )


def _pending_preflight_methods(
    args: argparse.Namespace,
    cohort: Cohort,
    classifier: TaxonomyClassifier,
) -> set[str]:
    """Backends needed by the pending fresh classifications and review retries."""
    if classifier.saved_first_pass is not None:
        return {classifier.second_pass}
    pending = {
        f"{row['run_id']}:{row['task_id']}:{int(row['repeat'])}"
        for row in cohort.rows
    }
    retries = set(getattr(args, "_resume_review_parents", {}) or {}) & pending
    methods = {classifier.second_pass} if retries else set()
    if pending - retries:
        methods.add(classifier.first_pass)
        methods.add(classifier.second_pass)
    methods.discard("off")
    return methods


def _preflight_classifier(
    classifier: TaxonomyClassifier,
    *,
    methods: set[str] | None = None,
) -> None:
    active = (
        methods
        if methods is not None
        else {classifier.first_pass, classifier.second_pass}
    )
    if "llm" in active and isinstance(classifier.provider, OpenAIProvider):
        _check_judge(classifier.provider)
    if "agentic" in active and classifier.agentic_provider is not None:
        from dataclasses import replace

        from .agentic_review import AgenticClassificationProvider, _preflight_harness

        agentic = classifier.agentic_provider
        if isinstance(agentic, AgenticClassificationProvider):
            resolved = _preflight_harness(agentic.harness, agentic.cli_path)
            if resolved != agentic.cli_path:
                classifier.agentic_provider = replace(agentic, cli_path=resolved)


def _check_judge(provider: OpenAIProvider) -> None:
    """Check the judge with one small request before downloading source data."""
    import dataclasses

    import openai

    from autofa.config import load_config
    from autofa.llm.endpoints import endpoint_setup_help

    probe = dataclasses.replace(
        provider,
        timeout_seconds=JUDGE_PROBE_TIMEOUT_SECONDS,
        max_retries=0,
    )
    question = {
        "type": "choice",
        "instructions": "Choose OK.",
        "criteria": {"OK": "The connection works.", "FAILED": "Something is wrong."},
    }
    try:
        probe.evaluate("The connection works.", {"probe": question})
    except ProviderError as exc:
        if isinstance(exc.__cause__, (TimeoutError, openai.APITimeoutError)):
            return
        raise ProviderError(
            f"judge {provider.model} is not usable: {exc}. "
            f"{endpoint_setup_help(load_config(), 'judge')}"
        ) from exc
    if not provider.supports_images:
        return
    try:
        probe.evaluate(
            "An image is attached.",
            {"probe": {**question, "instructions": "Choose OK if you can see the image."}},
            images=[_probe_image()],
        )
    except ProviderError as exc:
        if isinstance(exc.__cause__, (TimeoutError, openai.APITimeoutError)):
            return
        if image_input_is_unsupported(exc):
            provider.supports_images = False
            return
        raise ProviderError(
            f"judge {provider.model} image probe failed: {exc}"
        ) from exc


def _probe_image() -> dict[str, Any]:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    data = buffer.getvalue()
    return {
        "status": "available",
        "data_url": "data:image/png;base64," + base64.b64encode(data).decode(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _provider(
    args: argparse.Namespace, *, config: Any | None = None,
) -> OpenAIProvider:
    from autofa.config import load_config, load_dotenv
    from autofa.llm.endpoints import endpoint_for_config

    load_dotenv()
    params = dict(args.judge_params)
    api = params.pop("api", "chat_completions")
    configured = endpoint_for_config(config or load_config(), "judge", api=api)
    return OpenAIProvider(
        model=args.judge_model or configured.model,
        endpoint=configured.base_url,
        api_key_env=configured.api_key_env,
        request_options={**configured.request_options, **params},
        api=api,
        timeout_seconds=args.timeout,
        defer_api_key=bool(getattr(args, "_defer_judge_preflight", False)),
    )


def _classify_dataset(
    args: argparse.Namespace,
    cohort: Cohort,
    datasets: list[dict[str, Any]],
    api: DataApiClient,
    classifier: TaxonomyClassifier | None,
    judge: str,
    destination: Path,
) -> int:
    """Classify failed dataset rows."""
    from autofa.util.cancellation import Cancellation, cancellation_scope

    done = _resume_rollout_ids(args, destination, classifier)
    sources = _outcome_sources(cohort, api, timeout=DATA_API_TIMEOUT_SECONDS)
    source_identities = {
        str(source["run_id"]): _source_identity(source) for source in cohort.sources
    }
    counts: Counter[str] = Counter()
    submitted = 0
    completed = 0
    workers = max(1, args.workers)
    reward_filters = parse_reward_filters(args.reward or ["<=0"])
    remaining = args.limit if args.limit > 0 else None

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    cancellation = Cancellation()
    interrupted = False
    try:
        for run_id, batch in _dataset_batches(datasets, batch_size=max(64, workers * 4)):
            for item in batch:
                _require_classification_reward(item, run_id, reward_filters)
            items = [
                item
                for item in batch
                if _item_rollout_id(item, run_id) not in done
                and _rollout_in_shard(
                    _item_rollout_id(item, run_id),
                    args.shard_count,
                    args.shard_index,
                )
            ]
            if remaining is not None:
                items = items[:remaining]
                remaining -= len(items)
            if not items:
                if remaining == 0:
                    break
                continue

            futures = []
            for item in items:
                with cancellation_scope(cancellation):
                    context = copy_context()
                futures.append(
                    pool.submit(
                        context.run,
                        _process_dataset_item,
                        args,
                        item,
                        run_id,
                        sources[run_id],
                        source_identities[run_id],
                        classifier,
                    )
                )
            submitted += len(futures)
            def consume(future: Any, batch_futures: list[Any] = futures) -> None:
                nonlocal completed
                completed += 1
                row = future.result()
                counts[row["status"]] += 1
                if not args.prefetch_only:
                    _append_classification(
                        destination,
                        judge,
                        row,
                        args,
                        classifier=classifier,
                    )
                _print_progress(completed, row)
                _stop_on_fatal_error(row, batch_futures, cancellation)

            try:
                _consume_futures(futures, consume, cancellation=cancellation)
            except KeyboardInterrupt:
                interrupted = True
                cancellation.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            if remaining == 0:
                break
    finally:
        if not interrupted:
            pool.shutdown(wait=True)

    expected_status = "prepared" if args.prefetch_only else "ok"
    failed = submitted - counts[expected_status]
    summary = " ".join(f"{status}={n}" for status, n in sorted(counts.items()))
    action = "prefetched" if args.prefetch_only else "wrote"
    destination_text = "" if args.prefetch_only else f" to {destination}"
    _status(
        f"{action} {submitted}{destination_text}: {counts[expected_status]} succeeded, "
        f"{failed} failed" + (f" ({summary})" if summary else "")
    )
    return 1 if failed else 0


def _cached_evidence(
    args: argparse.Namespace,
    rollout_ids: Sequence[str],
    source_identities: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Split rollouts into usable cached evidence and the reason each other one is missing."""
    reward_filters = parse_reward_filters(args.reward or ["<=0"])
    cached: list[dict[str, Any]] = []
    missing: dict[str, str] = {}
    for rollout_id in rollout_ids:
        path = _cache_path(args.evidence_cache, rollout_id)
        if not path.is_file():
            missing[rollout_id] = "not in the evidence cache"
            continue
        try:
            prepared = _load_cache(
                path,
                rollout_id=rollout_id,
                max_characters=args.max_characters,
                source_identity_sha256=source_identities.get(rollout_id.split(":", 1)[0]),
            )
        except (OSError, ValueError) as exc:
            missing[rollout_id] = str(exc)
            continue
        if prepared.get("status") != "prepared" or _prepared_cache_needs_refresh(prepared):
            missing[rollout_id] = str(prepared.get("error") or "cached evidence is incomplete")
        elif not _prepared_reward_matches(prepared, reward_filters):
            missing[rollout_id] = "cached reward is outside the reward filter"
        else:
            cached.append(prepared)
    return cached, missing


def _classify_cached_rows(
    args: argparse.Namespace,
    cohort: Cohort,
    work_cohort: Cohort,
    classifier: TaxonomyClassifier | None,
    destination: Path,
) -> tuple[Cohort, int]:
    """Classify pending rows found in the evidence cache and return the rows still to prepare."""
    cached, _ = _cached_evidence(
        args,
        [f"{row['run_id']}:{row['task_id']}:{int(row['repeat'])}" for row in work_cohort.rows],
        {str(source["run_id"]): _source_identity(source) for source in work_cohort.sources},
    )
    if not cached:
        return work_cohort, 0
    _status(f"evidence cache: {len(cached)} of {work_cohort.count()} rollout(s) already prepared")
    status = 0 if args.prefetch_only else _classify_cached(args, classifier, destination, cached)
    found = {record["rollout_id"] for record in cached}
    rows = [
        row
        for row in work_cohort.rows
        if f"{row['run_id']}:{row['task_id']}:{int(row['repeat'])}" not in found
    ]
    return _cohort_rows(cohort, rows, limit=args.limit), status


def _classify_results_from_cache(
    args: argparse.Namespace, logs: ExitStack, *, config: Any | None = None,
) -> int:
    """Review saved results from cached evidence when their saved selection is unavailable."""
    destination = _classification_output(args.output, args.run, None)
    if not args.dry_run:
        _configure_logging(logs, destination)
    args._defer_judge_preflight = True
    classifier = None
    if not args.dry_run:
        classifier = (
            _classifier(args, config=config) if config is not None else _classifier(args)
        )
    if classifier is not None:
        _validate_output_profile(
            destination,
            classifier,
            resume=args.resume,
            parent_configuration=_saved_parent_configuration(args),
        )
    done = _resume_rollout_ids(args, destination, classifier)
    rollout_ids = [
        rollout_id
        for rollout_id in args._second_pass_parents
        if rollout_id not in done
        and _rollout_in_shard(rollout_id, args.shard_count, args.shard_index)
    ]
    if args.limit > 0:
        rollout_ids = rollout_ids[: args.limit]
    cached, missing = _cached_evidence(args, rollout_ids, {})
    for rollout_id, reason in missing.items():
        _status(f"warning: skipped {rollout_id}: {reason}", level=logging.WARNING)
    if args.dry_run:
        _status(f"{len(cached)} rollout(s) ready in the evidence cache, {len(missing)} missing")
        return 1 if missing else 0
    try:
        if cached and classifier is not None:
            _preflight_classifier(
                classifier,
                methods={classifier.second_pass},
            )
        status = _classify_cached(args, classifier, destination, cached) if cached else 0
    except RuntimeError:
        if destination.is_file():
            _write_manifest(args, destination, None, classifier)
        raise
    return _write_outputs(args, destination, max(status, int(bool(missing))), None, classifier)


def _classify_cached(
    args: argparse.Namespace,
    classifier: TaxonomyClassifier | None,
    destination: Path,
    prepared_records: list[dict[str, Any]],
) -> int:
    """Classify prepared evidence records without reading the sources."""
    from autofa.util.cancellation import Cancellation, cancellation_scope

    if classifier is None:
        raise RuntimeError("classifier is required to classify cached evidence")
    counts: Counter[str] = Counter()
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    cancellation = Cancellation()
    interrupted = False
    try:
        futures = []
        for prepared in prepared_records:
            with cancellation_scope(cancellation):
                context = copy_context()
            futures.append(
                pool.submit(
                    context.run,
                    classify_prepared,
                    classifier,
                    prepared,
                    parent=_parent_classification(
                        args,
                        str(prepared.get("rollout_id") or ""),
                    ),
                )
            )
        completed = 0

        def consume(future: Any) -> None:
            nonlocal completed
            completed += 1
            row = future.result()
            counts[row["status"]] += 1
            _append_classification(
                destination,
                "chat",
                row,
                args,
                classifier=classifier,
            )
            _print_progress(completed, row)
            _stop_on_fatal_error(row, futures, cancellation)

        try:
            _consume_futures(futures, consume, cancellation=cancellation)
        except KeyboardInterrupt:
            interrupted = True
            cancellation.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True)

    failed = len(prepared_records) - counts["ok"]
    summary = " ".join(f"{status}={n}" for status, n in sorted(counts.items()))
    _status(
        f"wrote {len(prepared_records)} from the evidence cache to {destination}: "
        f"{counts['ok']} succeeded, {failed} failed" + (f" ({summary})" if summary else "")
    )
    return 1 if failed else 0


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_identity(source: dict[str, Any]) -> str:
    """Hash source metadata that identifies the prepared data."""
    fields = (
        "kind",
        "run_id",
        "base_url",
        "store_id",
        "dataset_contract",
        "metadata_watermark",
    )
    return _json_sha256({key: source.get(key) for key in fields})


def _outcome_sources(
    cohort: Cohort,
    api: DataApiClient,
    *,
    timeout: float,
) -> dict[str, OutcomeSource]:
    from autofa.data.local_trace_store import resolve_store

    sources: dict[str, OutcomeSource] = {}
    for source in cohort.sources:
        run_id = source["run_id"]
        if source["kind"] == "local":
            sources[run_id] = LocalOutcomeSource(resolve_store(source["store_id"]))
            continue
        client = (
            api
            if api.base_url == source["base_url"]
            else DataApiClient(source["base_url"], timeout=timeout)
        )
        sources[run_id] = RemoteOutcomeSource(client, run_id)
    return sources


def _dataset_batches(datasets: list[dict[str, Any]], *, batch_size: int):
    for reference in datasets:
        dataset = Dataset([Path(reference["parquet_path"])])
        columns = dataset.resolve_columns([*dataset.resolve_columns(None), "reward"])
        for batch in dataset.iter_batches(batch_size=batch_size, columns=columns):
            yield reference["run_id"], batch


def _require_classification_reward(
    item: dict[str, Any],
    run_id: str,
    filters: list[tuple[str, float]],
) -> None:
    if not matches_reward_filters(item.get("reward"), filters):
        raise SourceDataError(
            f"classification input {_item_rollout_id(item, run_id)} "
            "is outside the reward filter"
        )


def _prepared_reward_matches(
    prepared: dict[str, Any],
    filters: list[tuple[str, float]],
) -> bool:
    incident = prepared.get("incident")
    provenance = incident.get("source_provenance") if isinstance(incident, dict) else {}
    provenance = provenance if isinstance(provenance, dict) else {}
    return matches_reward_filters(provenance.get("reward"), filters)


def _prepared_cache_needs_refresh(prepared: dict[str, Any]) -> bool:
    receipt = prepared.get("evidence_receipt") or {}
    verifier = receipt.get("verifier") or {}
    raw_outcome = receipt.get("raw_outcome") or {}
    return (
        isinstance(verifier.get("read_failures"), int)
        and verifier["read_failures"] > 0
    ) or raw_outcome.get("status") == "error"


def _process_dataset_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    run_id: str,
    source: OutcomeSource,
    source_identity: str,
    classifier: TaxonomyClassifier | None,
) -> dict[str, Any]:
    prepared = _prepare_dataset_item(args, item, run_id, source, source_identity)
    if args.prefetch_only:
        return prepared
    if classifier is None:
        raise RuntimeError("classifier is required unless --prefetch-only is set")
    return classify_prepared(
        classifier,
        prepared,
        parent=_parent_classification(args, str(prepared.get("rollout_id") or "")),
    )


def _parent_classification(
    args: argparse.Namespace,
    rollout_id: str,
) -> dict[str, Any] | None:
    for name in ("_second_pass_parents", "_resume_review_parents"):
        parents = getattr(args, name, None)
        if not isinstance(parents, dict):
            continue
        parent = parents.get(rollout_id)
        if isinstance(parent, dict):
            return parent
    return None


def _prepare_dataset_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    run_id: str,
    source: OutcomeSource,
    source_identity: str,
) -> dict[str, Any]:
    rollout_id = _item_rollout_id(item, run_id)
    source_sha256 = _json_sha256({"source": source_identity, "row": item})
    cache_path = (
        _cache_path(args.evidence_cache, rollout_id) if args.evidence_cache else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = _load_cache(
                cache_path,
                rollout_id=rollout_id,
                max_characters=args.max_characters,
                source_sha256=source_sha256,
            )
            reward_filters = parse_reward_filters(args.reward or ["<=0"])
            if (
                cached.get("status") == "prepared"
                and not _prepared_cache_needs_refresh(cached)
                and _prepared_reward_matches(cached, reward_filters)
            ):
                return cached
        except (OSError, ValueError):
            pass

    prepared = prepare_rollout(
        item,
        source,
        run_id=run_id,
        max_serialized_characters=args.max_characters,
    )
    prepared["source_sha256"] = source_sha256
    prepared["source_identity_sha256"] = source_identity
    if cache_path is not None:
        try:
            _write_cache(cache_path, prepared)
        except OSError as exc:
            return {
                **prepared,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": f"cannot write evidence cache {cache_path}: {exc}",
            }
    return prepared


def _item_rollout_id(item: dict[str, Any], run_id: str) -> str:
    return f"{run_id}:{item['problem_id']}:{int(item.get('run_idx') or 0)}"


def _judge_configuration(
    classifier: TaxonomyClassifier,
    method: str,
) -> dict[str, Any]:
    if method == "agentic":
        agentic = classifier.agentic_provider
        return {
            "requested_model": getattr(agentic, "model", None),
            "api": f"{getattr(agentic, 'harness', 'agentic')}_restricted",
            "configuration_sha256": (
                agentic.configuration_sha256() if agentic is not None else None
            ),
        }
    if method == "llm":
        provider = classifier.provider
        return {
            "requested_model": getattr(provider, "model", None),
            "api": getattr(provider, "api", None),
            "configuration_sha256": provider_configuration_sha256(provider),
        }
    return {}


def _output_token_limit(provider: OpenAIProvider | None) -> int | None:
    options = getattr(provider, "request_options", None) or {}
    return next(
        (options[key] for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
         if key in options),
        None,
    )


def _append_classification(
    destination: Path,
    judge: str,
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    classifier: TaxonomyClassifier,
) -> None:
    provider = classifier.provider
    agentic = classifier.agentic_provider
    parent_classification = dict(row.get("parent_classification") or {})
    agentic_request = (
        classifier.second_pass == "agentic"
        if classifier.saved_first_pass is not None
        else classifier.first_pass == "agentic"
    )
    primary_method = "agentic" if agentic_request else "llm"
    primary_configuration = _judge_configuration(classifier, primary_method)
    second_pass_configuration = _judge_configuration(
        classifier, classifier.second_pass
    )
    resumed_configuration = (
        second_pass_configuration if parent_classification else {}
    )
    reasoning_effort = (
        getattr(agentic, "effort", None)
        if agentic_request
        else effective_reasoning_effort(provider)
    )
    analysis = dict(row.get("analysis") or {})
    analysis.setdefault("taxonomy_id", classifier.taxonomy.taxonomy_id)
    analysis.setdefault("taxonomy_name", classifier.taxonomy.name)
    analysis.setdefault("taxonomy_version", classifier.taxonomy.version)
    analysis.setdefault("taxonomy_sha256", classifier.taxonomy.content_sha256)
    analysis.setdefault("protocol_version", PROTOCOL_VERSION)
    first_pass = dict(analysis.get("first_pass") or {})
    first_pass_configuration = _judge_configuration(classifier, classifier.first_pass)
    for key in ("requested_model", "api", "configuration_sha256"):
        if first_pass_configuration.get(key) is not None:
            first_pass.setdefault(key, first_pass_configuration[key])
    analysis["first_pass"] = first_pass
    analysis.setdefault(
        "second_pass",
        {
            "method": classifier.second_pass,
            "status": "skipped",
            "outcome": None,
            "usage": {},
        },
    )
    parent_configuration = _saved_parent_configuration(args)
    if parent_classification and parent_configuration:
        parent_classification["judge_configuration"] = parent_configuration
    append_classification(
        destination,
        classification_record(
            status=row["status"],
            judge_provider="agentic" if agentic_request else judge,
            classification_method=(
                "agentic_taxonomy_classification"
                if agentic_request
                else "llm_as_a_judge"
            ),
            analysis=analysis,
            rollout_id=row["rollout_id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            repeat=row["repeat"],
            lane=row.get("lane"),
            benchmark=row.get("benchmark"),
            model=row.get("model"),
            agent_ref=row.get("agent_ref"),
            digest_budget=row.get("digest_budget"),
            evidence_receipt=row.get("evidence_receipt"),
            classification_latency_ms=row.get("classification_latency_ms"),
            judge_request={
                "attempted": bool(row.get("judge_attempted")),
                **(
                    {
                        "resumed_pass": classifier.second_pass,
                        "resumed_requested_model": resumed_configuration[
                            "requested_model"
                        ],
                        "resumed_api": resumed_configuration["api"],
                        "resumed_configuration_sha256": resumed_configuration[
                            "configuration_sha256"
                        ],
                    }
                    if resumed_configuration
                    else {}
                ),
                "requested_model": primary_configuration["requested_model"],
                "api": primary_configuration["api"],
                "configuration_sha256": primary_configuration[
                    "configuration_sha256"
                ],
                **(
                    {
                        "second_pass_requested_model": second_pass_configuration[
                            "requested_model"
                        ],
                        "second_pass_api": second_pass_configuration["api"],
                        "second_pass_configuration_sha256": second_pass_configuration[
                            "configuration_sha256"
                        ],
                    }
                    if second_pass_configuration
                    else {}
                ),
                **(
                    {
                        "harness": getattr(agentic, "harness", None),
                        "configured_max_turns": getattr(agentic, "max_turns", None),
                    }
                    if agentic is not None
                    else {}
                ),
                "reasoning_effort": reasoning_effort,
                "reasoning_effort_source": (
                    "explicit"
                    if not agentic_request and "reasoning_effort" in dict(args.judge_params)
                    else (
                        "configured"
                        if not agentic_request
                        and (getattr(provider, "request_options", None) or {}).get(
                            "reasoning_effort"
                        )
                        else "judge_endpoint"
                        if agentic_request
                        else "endpoint_default_model_specific"
                    )
                ),
                "max_tokens": (
                    None if agentic_request else _output_token_limit(provider)
                ),
            },
            selection={
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
            },
            parent_classification=parent_classification,
            error_type=row.get("error_type"),
            error=row.get("error"),
        ),
    )


def _validate_shard_args(args: argparse.Namespace) -> None:
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be between 0 and shard-count - 1")
    if (
        args.shard_count > 1
        and not args.output
        and not args.prefetch_only
        and not args.dry_run
    ):
        raise ValueError(
            "sharded classification requires --output with a shard-specific filename"
        )


def _rollout_in_shard(rollout_id: str, shard_count: int, shard_index: int) -> bool:
    digest = hashlib.sha256(rollout_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count == shard_index


def _stop_on_fatal_error(
    row: dict[str, Any],
    futures: list[Any],
    cancellation: Any = None,
) -> None:
    if row.get("error_type") not in _FATAL_JUDGE_ERRORS:
        return
    if cancellation is not None:
        cancellation.cancel()
    for future in futures:
        future.cancel()
    raise RuntimeError(
        f"stopping: the judge rejected the request ({row['error_type']}); "
        "check the judge flags or Settings > LLM endpoints"
    )


def _consume_futures(
    futures: list[Any],
    consume: Any,
    *,
    cancellation: Any = None,
) -> None:
    """Consume completions and checkpoint every result complete at interruption."""
    handled: set[Any] = set()
    try:
        for future in as_completed(futures):
            handled.add(future)
            consume(future)
    except (KeyboardInterrupt, RuntimeError):
        if cancellation is not None:
            cancellation.cancel()
        for future in futures:
            future.cancel()
        for future in futures:
            if future in handled or future.cancelled() or not future.done():
                continue
            try:
                future.result()
            except (FutureCancelledError, asyncio.CancelledError):
                continue
            handled.add(future)
            try:
                consume(future)
            except RuntimeError:
                pass
        raise


def _print_progress(index: int, row: dict[str, Any]) -> None:
    detail = f" {row['error_type']}: {row['error']}" if row.get("error") else ""
    _status(f"[{index}] {row['status']} {row['rollout_id']}{detail}", flush=True)


def _cache_path(root: Path, rollout_id: str) -> Path:
    digest = hashlib.sha256(rollout_id.encode()).hexdigest()
    return root / digest[:2] / f"{digest}.json"


def _write_cache(path: Path, prepared: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_cache(path: Path) -> dict[str, Any]:
    prepared = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prepared, dict):
        raise ValueError(f"evidence cache must contain a JSON object: {path}")
    return prepared


def _load_cache(
    path: Path,
    *,
    rollout_id: str,
    max_characters: int,
    source_sha256: str | None = None,
    source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    prepared = _read_cache(path)
    if prepared.get("cache_version") != EVIDENCE_CACHE_VERSION:
        raise ValueError(
            f"evidence cache version mismatch: {path}; rebuild with {EVIDENCE_CACHE_VERSION}"
        )
    if prepared.get("rollout_id") != rollout_id:
        raise ValueError(f"evidence cache identity mismatch: {path}")
    if int(prepared.get("max_serialized_characters") or 0) != max_characters:
        raise ValueError(f"evidence cache budget mismatch: {path}; rebuild with {max_characters}")
    if not isinstance(prepared.get("source_sha256"), str) or not prepared["source_sha256"]:
        raise ValueError(f"evidence cache provenance is missing: {path}; rebuild it")
    if source_sha256 is not None and prepared.get("source_sha256") != source_sha256:
        raise ValueError(f"evidence cache source mismatch: {path}; rebuild from current source")
    if source_identity_sha256 is not None and not prepared.get("source_identity_sha256"):
        raise ValueError(f"evidence cache provenance is missing: {path}; rebuild it")
    if (
        source_identity_sha256 is not None
        and prepared.get("source_identity_sha256") != source_identity_sha256
    ):
        raise ValueError(f"evidence cache source identity mismatch: {path}; rebuild it")
    return prepared


def _saved_parent_configuration(args: argparse.Namespace) -> str | None:
    parents = getattr(args, "_second_pass_parents", None)
    if not isinstance(parents, dict) or not parents:
        return None
    configurations = {
        report._judge_configuration(parent)
        for parent in parents.values()
        if isinstance(parent, dict)
    }
    return next(iter(configurations)) if len(configurations) == 1 else None


def _saved_parent_sha256s(args: argparse.Namespace) -> dict[str, str] | None:
    parents = getattr(args, "_second_pass_parents", None)
    if not isinstance(parents, dict):
        return None
    return {
        rollout_id: classification_sha256(parent)
        for rollout_id, parent in parents.items()
        if isinstance(parent, dict)
    }


def _classification_profile(
    classifier: TaxonomyClassifier | None,
    *,
    parent_configuration: str | None = None,
) -> dict[str, Any] | None:
    if classifier is None:
        return None
    agentic_request = (
        classifier.second_pass == "agentic"
        if classifier.saved_first_pass is not None
        else classifier.first_pass == "agentic"
    )
    primary_configuration = _judge_configuration(
        classifier, "agentic" if agentic_request else "llm"
    )
    second_pass_configuration = _judge_configuration(
        classifier, classifier.second_pass
    )
    agentic = classifier.agentic_provider
    agentic_configuration = (
        agentic.configuration()
        if agentic is not None and callable(getattr(agentic, "configuration", None))
        else None
    )
    return {
        "taxonomy_sha256": classifier.taxonomy.content_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "first_pass": classifier.first_pass,
        "requested_model": primary_configuration["requested_model"],
        "configuration_sha256": primary_configuration["configuration_sha256"],
        "second_pass": classifier.second_pass,
        "second_pass_configuration_sha256": second_pass_configuration.get(
            "configuration_sha256"
        ),
        "agentic_configuration": agentic_configuration,
        "parent_configuration": parent_configuration,
    }


def _matches_judge_configuration(
    record: dict[str, Any],
    profile: dict[str, Any],
    field: str,
) -> bool:
    judge = record.get("judge") or {}
    recorded = judge.get(field)
    expected = profile.get(field)
    if recorded == expected:
        return True
    configuration = profile.get("agentic_configuration")
    if not isinstance(configuration, dict) or expected != _configuration_sha256(configuration):
        return False
    prior_limit = judge.get("configured_max_turns")
    if prior_limit is not None and type(prior_limit) is not int:
        return False
    legacy_configuration = {
        key: value
        for key, value in configuration.items()
        if key != "tool_budget_semantics"
    }
    compatible = []
    for limit in (prior_limit, None):
        compatible.append({
            **legacy_configuration,
            "configured_max_turns": prior_limit,
            "tool_call_limit": limit,
        })
    return any(recorded == _configuration_sha256(candidate) for candidate in compatible)


def _configuration_sha256(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _matches_classification_profile(
    record: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    judge = record.get("judge") or {}
    return (
        record.get("taxonomy_sha256") == profile["taxonomy_sha256"]
        and record.get("protocol_version") == profile["protocol_version"]
        and judge.get("requested_model") == profile["requested_model"]
        and _matches_judge_configuration(
            record,
            profile,
            "configuration_sha256",
        )
        and _record_first_pass_method(record) == profile.get("first_pass", "llm")
        and _matches_judge_configuration(
            record,
            profile,
            "second_pass_configuration_sha256",
        )
        and (
            profile.get("second_pass") is None
            or _record_second_pass_method(record) == profile["second_pass"]
        )
        and (
            profile.get("parent_configuration") is None
            or (record.get("parent_classification") or {}).get(
                "judge_configuration"
            )
            == profile["parent_configuration"]
        )
    )


def _record_first_pass_method(record: dict[str, Any]) -> str:
    first_pass = record.get("first_pass")
    if isinstance(first_pass, dict) and first_pass.get("method") in {"llm", "agentic"}:
        return str(first_pass["method"])
    return (
        "agentic"
        if record.get("classification_method") == "agentic_taxonomy_classification"
        else "llm"
    )


def _record_second_pass_method(record: dict[str, Any]) -> str | None:
    second_pass = record.get("second_pass")
    if isinstance(second_pass, dict) and second_pass.get("method") in {
        "off",
        "llm",
        "agentic",
    }:
        return str(second_pass["method"])
    return None


def _ok_rollout_ids(
    destination: Path,
    profile: dict[str, Any],
    *,
    parent_sha256s: dict[str, str] | None = None,
) -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for record in read_classifications(destination, tolerate_truncated_final=True):
        if not _matches_classification_profile(record, profile):
            continue
        rollout_id = str((record.get("source") or {}).get("rollout_id"))
        latest[rollout_id] = record
    completed = set()
    for rollout_id, record in latest.items():
        if record.get("status") != "ok":
            continue
        if parent_sha256s is not None and (
            (record.get("parent_classification") or {}).get("record_sha256")
            != parent_sha256s.get(rollout_id)
        ):
            continue
        second_pass = record.get("second_pass")
        if (
            profile.get("second_pass") != "off"
            and isinstance(second_pass, dict)
            and second_pass.get("status") not in {"completed", "skipped"}
        ):
            continue
        completed.add(rollout_id)
    return completed


def _review_failed_parents(
    destination: Path,
    profile: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if profile is None or not destination.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for record in read_classifications(destination, tolerate_truncated_final=True):
        if not _matches_classification_profile(record, profile):
            continue
        rollout_id = str((record.get("source") or {}).get("rollout_id") or "")
        if rollout_id:
            latest[rollout_id] = record
    return {
        rollout_id: {
            **record,
            "_results_path": str(destination.resolve()),
        }
        for rollout_id, record in latest.items()
        if record.get("status") == "review_failed" and has_usable_first_pass(record)
    }


def _resume_rollout_ids(
    args: argparse.Namespace,
    destination: Path,
    classifier: TaxonomyClassifier | None,
) -> set[str]:
    parent_configuration = _saved_parent_configuration(args)
    parent_sha256s = _saved_parent_sha256s(args)
    profile = _classification_profile(
        classifier,
        parent_configuration=parent_configuration,
    )
    if profile is None:
        return set()
    done = (
        _ok_rollout_ids(destination, profile, parent_sha256s=parent_sha256s)
        if args.resume
        else set()
    )
    for source in args.resume_from:
        if not source.exists():
            raise ValueError(f"--resume-from path does not exist: {source}")
        explicit = source.is_file()
        if explicit:
            paths = [source]
        elif source.is_dir():
            try:
                paths = sorted(source.rglob("*.jsonl"))
            except OSError as exc:
                raise ValueError(f"cannot read --resume-from directory {source}: {exc}") from exc
        else:
            raise ValueError(f"--resume-from path is not a file or directory: {source}")
        for path in paths:
            try:
                done.update(
                    _ok_rollout_ids(path, profile, parent_sha256s=parent_sha256s)
                )
            except (OSError, ValueError) as exc:
                if explicit:
                    raise ValueError(f"cannot read --resume-from file {path}: {exc}") from exc
                _status(f"warning: skipped --resume-from file {path}: {exc}", level=logging.WARNING)
    return done


def _validate_output_profile(
    destination: Path,
    classifier: TaxonomyClassifier,
    *,
    resume: bool,
    parent_configuration: str | None = None,
) -> None:
    """Keep one append-only result file tied to one taxonomy/judge profile."""
    profile = _classification_profile(
        classifier,
        parent_configuration=parent_configuration,
    )
    if profile is None:
        return
    records = read_classifications(
        destination,
        tolerate_truncated_final=resume,
        repair_truncated_final=resume,
    )
    if any(not _matches_classification_profile(record, profile) for record in records):
        raise ValueError(
            f"{destination} contains a different taxonomy, judge, or second-pass profile; "
            "use a separate --output file"
        )


def _classification_output(output: str | None, run_id: str | None, cohort: Cohort | None) -> Path:
    if output:
        return Path(output)
    if run_id:
        from autofa.config import runs_root

        return runs_root() / run_id / "taxonomy_classifications.jsonl"
    if cohort is not None:
        return cohorts_root() / cohort.id / "taxonomy_classifications.jsonl"
    return Path("taxonomy_classifications.jsonl")


def _write_outputs(
    args: argparse.Namespace,
    destination: Path,
    status: int,
    cohort: Cohort | None,
    classifier: TaxonomyClassifier | None,
) -> int:
    if not destination.is_file():
        return status
    if classifier is None:
        raise RuntimeError("classifier is required to write result provenance")
    _write_manifest(
        args,
        destination,
        cohort,
        classifier,
    )
    try:
        path = report.write_report(
            [destination], destination.with_suffix(".html"), taxonomy=args.taxonomy
        )
    except (OSError, ValueError) as exc:
        _status(f"warning: report not written: {exc}", level=logging.WARNING)
        return status
    command = ["autofa", "classify", "report", str(destination)]
    if args.taxonomy:
        command.extend(["--taxonomy", str(args.taxonomy)])
    _status(f"Report: {path}  (regenerate: {shlex.join(command)})")
    return status


def _write_manifest(
    args: argparse.Namespace,
    destination: Path,
    cohort: Cohort | None,
    classifier: TaxonomyClassifier,
) -> None:
    parent_results = {
        str(parent["_results_path"])
        for parent in (getattr(args, "_second_pass_parents", None) or {}).values()
        if parent.get("_results_path")
    }
    path = write_manifest(
        destination,
        cohort=cohort,
        classifier=classifier,
        selection={
            **(
                {
                    "runs_where": cohort.runs_where,
                    "where": cohort.where,
                    "latest": cohort.latest,
                    "per_run": cohort.per_run,
                    "sample": cohort.sample,
                    "seed": cohort.seed,
                }
                if cohort
                else {}
            ),
            "limit": args.limit,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
        workers=max(1, args.workers),
        parent_results=sorted(parent_results),
        workflow_config=getattr(args, "_workflow_config", None),
    )
    _status(f"Manifest: {path}")


def _print_taxonomy_error(error: TaxonomyError, path: str | None) -> None:
    source = Path(path).expanduser() if path else DEFAULT_TAXONOMY_PATH
    _status(f"error: invalid taxonomy {source}: {error}", level=logging.ERROR)
    _status(f"Example: {DEFAULT_TAXONOMY_PATH}", level=logging.ERROR)
