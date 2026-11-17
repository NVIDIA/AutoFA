# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset selection wizard with arrow-key navigation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from autofa.data.efficiency_support import check_local_sources, check_references
from autofa.data.gym_adapter import load_gym_metadata, materialize_eval_source
from autofa.data.parquet_store import infer_models_benchmarks
from autofa.data.preparation import PreparedData
from autofa.data.reference_data import (
    data_api_location,
    make_reference_data,
    normalize_reference_data,
    summarize_source_harness,
)
from autofa.data.remote_api import DEFAULT_DATA_API_BASE_URL
from autofa.data.sources import (
    EvalSource,
    discover_eval_sources,
    discover_parquets,
    recognized_log_folder,
)
from autofa.ui.errors import set_action_stage
from autofa.ui.menus import (
    MenuItem,
    browse_filesystem,
    is_menu_back,
    prepare_step,
    select_items_multi,
    select_option,
    select_paths_multi,
)
from autofa.ui.theme import GREEN
from autofa.wizard.remote_runs import pick_and_download_remote_run

console = Console()

FILE_GLOB = "*.{parquet,jsonl}"
DatasetSelection = PreparedData


def run_dataset_wizard(  # noqa: C901
    default_folder: str | None = None,
    *,
    inputs: list[str] | None = None,
    api_run_ids: list[str] | None = None,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    tasks: list[str] | None = None,
    rewards: list[str] | None = None,
    runs_where: str = "",
    where: str = "",
    latest: int = 0,
    per_run: int = 0,
    limit: int = 0,
    sample: str = "first",
    seed: int = 0,
    judge_view: str | None = None,
    group: str = "all",
    labels: list[str] | None = None,
    evidence: list[str] | None = None,
    list_taxonomy: bool = False,
    selection_id: str | None = None,
    api_run_id: str | None = None,
    api_base_url: str | None = None,
    api_name: str = "EvalRDL remote server",
    refresh_api: bool = False,
    efficiency_only: bool = False,
) -> DatasetSelection | None:
    """Pick eval logs (local parquet/Gym/JSONL or EvalRDL remote server API) and filters.

    When ``api_run_id`` is set, skip the source/browser steps and download that
    remote run's AutoFA export directly.
    """
    from autofa.services import validate_service_address

    base_url = validate_service_address(
        DEFAULT_DATA_API_BASE_URL if api_base_url is None else api_base_url,
        name="EvalRDL remote server",
    )

    explicit = any((
        inputs,
        api_run_ids,
        models,
        benchmarks,
        tasks,
        rewards,
        runs_where,
        where,
        latest,
        per_run,
        limit,
        sample != "first",
        seed,
        judge_view,
        group != "all",
        labels,
        evidence,
        list_taxonomy,
    ))
    if explicit:
        from autofa.data.cohort import (
            load_cohort,
            prepare_data,
            refine_cohort,
            resolve_inputs,
            select_cohort,
        )
        from autofa.wizard import remote_runs

        api = remote_runs.DataApiClient(base_url)
        input_kind, resolved = resolve_inputs(inputs or [])
        taxonomy_filters = bool(
            judge_view or group != "all" or labels or evidence or list_taxonomy
        )
        if taxonomy_filters and input_kind != "taxonomy":
            raise ValueError("taxonomy filters require taxonomy results passed with --input")
        if input_kind in {"selection", "taxonomy"} and any((
            api_run_ids,
            models,
            benchmarks,
            runs_where,
            latest,
            where,
        )):
            raise ValueError(
                f"{input_kind} input cannot be combined with run or advanced WHERE filters"
            )
        if input_kind == "local":
            from autofa.data.local_import import reference_for_import
            from autofa.data.sources import discover_eval_sources

            discovered = [
                source
                for value in resolved
                for source in discover_eval_sources(Path(value))
            ]
            check_local_sources(discovered)
            legacy = [
                source
                for source in discovered
                if source.kind != "evalrdl" and reference_for_import(source.path) is None
            ]
            if legacy:
                if len(legacy) != len(discovered):
                    raise ValueError("Do not mix managed EvalRDL and legacy local inputs")
                if any((
                    api_run_ids,
                    models,
                    benchmarks,
                    tasks,
                    rewards,
                    runs_where,
                    where,
                    latest,
                    per_run,
                    limit,
                    sample != "first",
                    seed,
                )):
                    raise ValueError(
                        "Rollout filters require managed EvalRDL data; prepare this input first"
                    )
                prepared_paths = _prepare_local_sources(
                    discovered,
                    efficiency_only=efficiency_only,
                )
                if prepared_paths is None:
                    return None
                return _finish_selection(
                    prepared_paths,
                    source_paths=[str(Path(value).resolve()) for value in resolved],
                    source_kind=discovered[0].kind if len(discovered) == 1 else "parquet",
                )
        if input_kind == "selection":
            cohort = refine_cohort(
                load_cohort(resolved[0]),
                tasks=tasks or [],
                rewards=rewards or [],
                per_run=per_run,
                sample=sample,
                seed=seed,
                limit=limit,
                api=api,
            )
        elif input_kind == "taxonomy":
            from autofa.taxonomy.groups import (
                classification_facets,
                cohort_from_classifications,
            )

            paths = [Path(value) for value in resolved]
            if list_taxonomy:
                facets = classification_facets(paths)
                console.print("Judges: " + ", ".join(facets["judges"]))
                console.print("Labels:")
                for code, label in sorted(facets["labels"].items()):
                    console.print(f"  {code}: {label}")
                console.print("Evidence: " + ", ".join(facets["evidence"]))
                return None
            cohort = cohort_from_classifications(
                paths,
                judge=judge_view,
                group=group,
                labels=labels or [],
                evidence=evidence or [],
            )
            cohort = refine_cohort(
                cohort,
                tasks=tasks or [],
                rewards=rewards or [],
                per_run=per_run,
                sample=sample,
                seed=seed,
                limit=limit,
            )
        else:
            cohort = select_cohort(
                api_runs=api_run_ids or [],
                folders=resolved if input_kind == "local" else [],
                models=models or [],
                benchmarks=benchmarks or [],
                tasks=tasks or [],
                rewards=rewards or [],
                runs_where=runs_where,
                where=where,
                latest=latest,
                per_run=per_run,
                limit=limit,
                sample=sample,
                seed=seed,
                api=api,
                refresh=refresh_api,
                efficiency_only=efficiency_only,
            )
        if cohort.taxonomy_filter:
            from autofa.taxonomy.groups import add_success_controls

            cohort = add_success_controls(cohort, api=api, activity=console.print)
        return prepare_data(
            cohort,
            api=api,
            activity=console.print,
            efficiency_only=efficiency_only,
            refresh=refresh_api,
        )

    if selection_id:
        from autofa.data.cohort import load_cohort, prepare_data
        from autofa.taxonomy.groups import add_success_controls
        from autofa.wizard import remote_runs

        cohort = load_cohort(selection_id)
        remote_sources = [
            source for source in cohort.sources if source.get("kind") == "remote"
        ]
        api = None
        if remote_sources:
            expected = base_url.rstrip("/")
            if any(
                str(source.get("base_url") or "").rstrip("/") != expected
                for source in remote_sources
            ):
                raise SystemExit(
                    "Saved selection points to a different Data API; "
                    "update data_api_base_url explicitly before reusing it"
                )
            api = remote_runs.DataApiClient(base_url)
        if cohort.taxonomy_filter:
            cohort = add_success_controls(
                cohort,
                api=api,
                activity=console.print,
            )
        return prepare_data(
            cohort,
            api=api,
            activity=console.print,
            efficiency_only=efficiency_only,
            refresh=refresh_api,
        )

    if api_run_id:
        from autofa.data.cohort import prepare_data, select_cohort
        from autofa.wizard import remote_runs

        api = remote_runs.DataApiClient(base_url)
        return _finish_prepared_selection(
            prepare_data(
                select_cohort(api_runs=[api_run_id], api=api),
                api=api,
                activity=console.print,
                efficiency_only=efficiency_only,
                refresh=refresh_api,
            )
        )

    while True:
        prepare_step()
        origin = select_option(
            "Dataset source",
            [
                MenuItem("Local eval logs (NeMo Gym / Harbor)", "local"),
                *([MenuItem(api_name, "remote")] if base_url else []),
                MenuItem("Taxonomy classification results", "taxonomy"),
            ],
            allow_cancel=True,
        )
        if origin is None:
            return None

        if origin.value == "taxonomy":
            from autofa.data.cohort import _pick_taxonomy_cohort, prepare_data
            from autofa.taxonomy.groups import add_success_controls
            from autofa.wizard import remote_runs

            cohort = _pick_taxonomy_cohort()
            if cohort is None:
                continue
            api = remote_runs.DataApiClient(base_url)
            return prepare_data(
                add_success_controls(cohort, api=api, activity=console.print),
                api=api,
                activity=console.print,
                efficiency_only=efficiency_only,
                refresh=refresh_api,
            )

        if origin.value == "remote":
            from autofa.data.cohort import prepare_data, select_cohort
            from autofa.wizard import remote_runs

            api = remote_runs.DataApiClient(base_url)
            remote = remote_runs.pick_remote_runs(api)
            if remote is None:
                continue
            selection = _finish_prepared_selection(
                prepare_data(
                    select_cohort(
                        api_runs=[str(run["run_id"]) for run in remote],
                        api=api,
                    ),
                    api=api,
                    activity=console.print,
                    efficiency_only=efficiency_only,
                    refresh=refresh_api,
                ),
            )
            return selection

        local = _run_local_wizard(
            default_folder=default_folder,
            efficiency_only=efficiency_only,
        )
        if local is None:
            continue  # Esc from filesystem browser → dataset source
        return local


def reuse_reference_data(
    reference_data: list[dict[str, str]] | tuple[dict[str, str], ...],
    *,
    allowed_base_url: str | None = None,
    refresh_api: bool = False,
    efficiency_only: bool = False,
) -> DatasetSelection | None:
    """Reopen managed local stores and rematerialize remote runs attached to a report."""
    for reference in reference_data:
        if not isinstance(reference, dict) or not normalize_reference_data([reference]) or (
            reference.get("kind") != "local_evalrdl" and data_api_location(reference) is None
        ):
            raise SystemExit(
                "Saved Data API provenance is incomplete; refusing partial reuse"
            )
    references = normalize_reference_data(reference_data, include_local_paths=False)
    if not references:
        raise SystemExit("Saved Data API references are missing or invalid")
    check_references(references)
    downloads = []
    merged_references: list[dict[str, str]] = []
    for reference in references:
        if reference.get("kind") == "local_evalrdl":
            import json

            from autofa.data.local_import import (
                LocalImport,
                materialize_local_traces,
                reference_for_import,
            )
            from autofa.data.local_trace_store import resolve_store
            from autofa.data.source_errors import SourceDataError

            try:
                store = resolve_store(reference["store_id"])
            except ValueError as exc:
                raise SystemExit(f"Saved local trace store is unavailable: {exc}") from exc
            try:
                # Validates the store manifest, including its required dataset_kind.
                actual = reference_for_import(store / "dataset.parquet")
            except SourceDataError as exc:
                raise SystemExit(str(exc)) from exc
            if actual is None or actual["run_id"] != reference["run_id"]:
                raise SystemExit("Saved local reference and imported run identity disagree")
            manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
            parquet = store / "dataset.parquet"
            dataset_kind = str(manifest["dataset_kind"])
            if not efficiency_only and dataset_kind == "compact":
                from autofa.config import autofa_home

                parquet = materialize_local_traces(
                    store,
                    parquet,
                    autofa_home() / "cache" / "local-traces" / store.name
                    / "dataset.traces.parquet",
                )
                dataset_kind = "rich"
            imported = LocalImport(
                parquet,
                reference["store_id"],
                actual["run_id"],
                actual["source_harness"],
                dataset_kind,
            )
            downloads.append(imported)
            merged_references.append(imported.reference_data())
            continue
        location = data_api_location(reference)
        assert location is not None
        base_url, run_id = location
        if (
            allowed_base_url is not None
            and base_url.rstrip("/") != allowed_base_url.rstrip("/")
        ):
            raise SystemExit(
                "Saved provenance points to a different Data API; "
                "update data_api_base_url explicitly before reusing it"
            )
        selected = pick_and_download_remote_run(
            base_url=base_url,
            run_id=run_id,
            force=refresh_api,
            efficiency_only=efficiency_only,
        )
        if not selected:
            raise SystemExit(f"Could not rematerialize referenced Data API run {run_id}")
        download = selected[0]
        downloads.append(download)
        harness = (
            reference.get("source_harness")
            or download.source_harness
            or "unknown"
        )
        merged_references.append(
            make_reference_data(
                run_id=run_id,
                export_url=reference["export_url"],
                source_harness=harness,
                parquet_path=download.parquet_path,
            )
        )
    if len(downloads) != len(references):
        raise SystemExit(
            "Not every referenced Data API run was materialized; refusing partial reuse"
        )
    selection = _finish_selection(
        [item.parquet_path for item in downloads],
        source_paths=[reference.get("export_url") or reference["parquet_path"]
                      for reference in merged_references],
        source_kind=("remote_run" if all("export_url" in ref for ref in merged_references)
                     else "evalrdl"),
        source_harness=summarize_source_harness(
            reference.get("source_harness") for reference in merged_references
        ),
        reference_data=merged_references,
    )
    return selection


def pick_local_sources(
    *, default_folder: str | None = None, raw_only: bool = False,
) -> tuple[str, list[EvalSource]] | None:
    """Select local eval sources without importing them."""
    prepare_step()
    if raw_only:
        selection_hint = "a Gym or Harbor run folder, or [bold]Enter[/bold] one .jsonl file"
        title = "Select a run folder (Gym / Harbor) or a .jsonl file"
        file_glob = "*.jsonl"
        expected = "a supported harness run folder (Gym / Harbor) or a .jsonl file"
    else:
        selection_hint = (
            "every parquet under it, a Gym or Harbor run folder, or "
            "[bold]Enter[/bold] one .parquet / .jsonl file"
        )
        title = "Select a run folder (Gym / Harbor) or a .parquet / .jsonl file"
        file_glob = FILE_GLOB
        expected = "a supported harness run folder (Gym / Harbor), *.parquet, or a .jsonl file"
    console.print(
        Panel(
            f"[bold {GREEN}]Dataset[/]\n"
            f"[dim][bold]↑/↓[/bold] navigate · [bold]Enter[/bold] opens a folder · choose "
            f"[bold {GREEN}][ USE THIS FOLDER ][/] for {selection_hint} · "
            f"[bold]Esc[/bold] back[/dim]",
            border_style=GREEN,
        )
    )

    start = Path(default_folder).expanduser() if default_folder else Path.home() / "Projects"
    if not start.exists():
        start = Path.home()

    chosen = browse_filesystem(
        start,
        title=title,
        select_files=True,
        select_dirs=True,
        file_glob=file_glob,
        folder_description=recognized_log_folder,
        clear_screen=False,
    )
    if chosen is None:
        return None

    root = Path(chosen)
    sources = discover_eval_sources(root)
    if raw_only:
        sources = [source for source in sources if source.kind != "parquet"]
    if not sources:
        raise SystemExit(f"No eval logs under {root}\nExpected {expected}.")

    source_selection = _pick_sources(root, sources)
    if source_selection is None:
        return None
    return source_selection


def _run_local_wizard(
    *,
    default_folder: str | None = None,
    efficiency_only: bool = False,
) -> DatasetSelection | None:
    """Select and prepare local parquet, Gym or JSONL data."""
    source_selection = pick_local_sources(default_folder=default_folder)
    if source_selection is None:
        return None
    source_kind, selected_sources = source_selection
    check_local_sources(selected_sources)
    source_paths = [str(s.path.resolve()) for s in selected_sources]

    selected = _prepare_local_sources(
        selected_sources,
        efficiency_only=efficiency_only,
    )
    if selected is None:
        return None

    if not selected:
        raise SystemExit("No parquet produced from selected sources")

    console.print(f"[dim]Canonical parquet:[/dim] {len(selected)} file(s)")
    for p in selected[:5]:
        console.print(f"[dim]  · {p}[/dim]")
    if len(selected) > 5:
        console.print(f"[dim]  · … +{len(selected) - 5} more[/dim]")

    from autofa.data.local_import import reference_for_import

    references = [ref for path in selected if (ref := reference_for_import(path))]
    if len(references) == len(selected):
        from autofa.data.cohort import prepare_data, select_cohort

        choice = _finish_selection(
            selected,
            source_paths=source_paths,
            source_kind=source_kind,
            source_harness=summarize_source_harness(
                reference.get("source_harness") for reference in references
            ),
            reference_data=references,
        )
        if choice is None:
            return None
        prepared = prepare_data(
            select_cohort(
                folders=selected,
                models=[] if choice.model_id == "*" else [choice.model_id],
                benchmarks=choice.benchmarks,
            ),
            activity=console.print,
            efficiency_only=efficiency_only,
        )
        prepared.model_id = choice.model_id
        prepared.benchmarks = choice.benchmarks
        prepared.source_paths = choice.source_paths
        prepared.source_kind = choice.source_kind
        return prepared
    return _finish_selection(
        selected,
        source_paths=source_paths,
        source_kind=source_kind,
        source_harness=summarize_source_harness([
            _local_source_harness(selected_sources),
            *(ref.get("source_harness") for ref in references),
        ]),
        reference_data=references,
    )


def _prepare_local_sources(
    sources: list[EvalSource],
    *,
    efficiency_only: bool = False,
) -> list[Path] | None:
    """Import each source, offering explicit labels for missing run metadata."""
    import pyarrow.parquet as pq

    from autofa.data.local_import import import_local_run, reference_for_import

    selected: list[Path] = []
    for index, source in enumerate(sources, 1):
        with Status(
            f"[{GREEN}]Preparing dataset ({index}/{len(sources)})…[/{GREEN}]",
            console=console, spinner="dots", spinner_style=GREEN,
        ):
            paths = _materialize_sources(
                [source],
                efficiency_only=efficiency_only,
            )
        # Relabeling creates a separate cached import without modifying Parquet inputs.
        if source.kind in {"evalrdl", "gym_dir", "jsonl"} and len(paths) == 1:
            reference = reference_for_import(paths[0])
            if reference:
                runs = pq.read_table(
                    paths[0].parent / "tables" / "runs.parquet",
                    columns=["run_id", "model", "benchmark"],
                ).to_pylist()
                run = next(row for row in runs if row["run_id"] == reference["run_id"])
                missing = [
                    field for field, column in (("model_id", "model"), ("benchmark", "benchmark"))
                    if not str(run.get(column) or "").strip()
                    or str(run[column]).strip().casefold() == "unknown"
                ]
                if missing:
                    overrides = _prompt_local_metadata(source.path, missing)
                    if overrides is None:
                        return None
                    if overrides:
                        with Status(
                            f"[{GREEN}]Preparing labeled dataset…[/{GREEN}]",
                            console=console, spinner="dots", spinner_style=GREEN,
                        ):
                            paths = [
                                import_local_run(
                                    source.path,
                                    **overrides,
                                    efficiency_only=efficiency_only,
                                ).parquet_path
                            ]
        selected.extend(paths)
    return list(dict.fromkeys(selected))


def _prompt_local_metadata(source: Path, missing: list[str]) -> dict[str, str] | None:
    """Optional source-specific labels; blank keeps unknown and Esc goes back."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.processors import BeforeInput

    from autofa.ui.pt_app import run_app
    from autofa.ui.theme import MENU_STYLE

    prepare_step()
    labels = {"model_id": "Tested model", "benchmark": "Benchmark"}
    controls = {
        field: BufferControl(
            buffer=Buffer(multiline=False),
            input_processors=[BeforeInput(f"{labels[field]} (optional): ")],
        )
        for field in missing
    }
    result: dict[str, str] | None = None
    bindings = KeyBindings()

    @bindings.add("enter")
    def accept(event):
        nonlocal result
        current = event.app.layout.current_control
        position = list(controls.values()).index(current)
        if position + 1 < len(controls):
            event.app.layout.focus(list(controls.values())[position + 1])
            return
        result = {
            field: control.buffer.text.strip() for field, control in controls.items()
            if control.buffer.text.strip()
        }
        event.app.exit()

    @bindings.add("tab")
    @bindings.add("s-tab")
    def move(event):
        controls_list = list(controls.values())
        position = controls_list.index(event.app.layout.current_control)
        offset = -1 if event.key_sequence[0].key == "s-tab" else 1
        event.app.layout.focus(controls_list[(position + offset) % len(controls_list)])

    @bindings.add("escape")
    @bindings.add("c-c")
    def back(event):
        event.app.exit()

    header = FormattedTextControl([
        ("class:title", "Optional run metadata\n"),
        ("", f"{source}\n\n"),
        ("class:hint", "These names are missing from the run. Enter names you know, "
         "or leave them blank to keep unknown.\n\n"),
    ])
    footer = FormattedTextControl([
        ("class:hint", "\nEnter next / continue · Tab switch field · Esc back"),
    ])
    app = Application(
        layout=Layout(HSplit([
            Window(header, dont_extend_height=True),
            *(Window(control, height=1) for control in controls.values()),
            Window(footer, dont_extend_height=True),
        ]), focused_element=next(iter(controls.values()))),
        key_bindings=bindings, style=MENU_STYLE, full_screen=False,
    )
    run_app(app)
    return result


def _finish_selection(
    selected: list[Path],
    *,
    source_paths: list[str],
    source_kind: str,
    source_harness: str = "unknown",
    reference_data: list[dict[str, str]] | None = None,
    selection_id: str | None = None,
) -> DatasetSelection | None:
    """Preserve picked remote runs; offer model/benchmark filters for local files."""
    preserve_runs = source_kind == "remote_run"
    set_action_stage("scan dataset", parquet_paths=[str(path) for path in selected])
    with Status(
        f"[{GREEN}]Scanning model_id / benchmark "
        f"({len(selected)} file(s))…[/{GREEN}]",
        console=console,
        spinner="dots",
        spinner_style=GREEN,
    ):
        meta = infer_models_benchmarks(selected)

    models = [m for m, _ in meta["models"]]
    benches = list(meta["benchmarks"])  # (name, count)
    n_rows = int(meta.get("n_rows") or 0)
    n_fail = int(meta.get("n_fail") or 0)

    console.print(
        f"[dim]Found {len(models)} model(s), {len(benches)} benchmark(s), "
        f"{sum(n for _, n in benches) or n_rows:,} labeled rows[/dim]"
    )
    if n_rows > 0 and n_fail == 0:
        console.print(
            "[yellow]Warning:[/yellow] 0 failures in this corpus "
            f"({n_rows} rows all pass). Discovery needs failed examples — "
            "pick a dump that includes fails, or expect empty / weak detectors."
        )

    if len(models) > 1 and preserve_runs:
        model_id = "*"
        console.print(f"[dim]models:[/dim] all {len(models)} in the selected rollout sets")
    elif len(models) > 1:
        choice = select_option(
            "Select model_id",
            [MenuItem(f"All models ({len(models)})", "*")]
            + [MenuItem(m, m) for m in models],
            allow_back=True,
        )
        if is_menu_back(choice) or choice is None:
            return None
        model_id = str(choice.value)
    elif models:
        model_id = models[0]
        console.print(f"[dim]model_id:[/dim] {model_id}")
    else:
        model_id = selected[0].stem if selected[0].is_file() else selected[0].name
        console.print(f"[dim]model_id not in columns — using {model_id}[/dim]")

    if benches:
        if len(benches) == 1:
            benchmarks = [benches[0][0]]
            console.print(f"[dim]benchmark:[/dim] {benchmarks[0]}")
            _warn_sensitive_benchmarks(benchmarks)
        else:
            if preserve_runs:
                # Preserve the selected corpus and concrete benchmark metadata.
                benchmarks = [b for b, _ in benches]
            else:
                picked = select_items_multi(
                    [MenuItem(f"{b}   ({n:,} rows)", b) for b, n in benches],
                    title=f"Select benchmarks ({len(benches)} found)",
                    empty_means_all=True,
                    confirm=True,
                    confirm_noun="benchmarks",
                )
                if picked is None:
                    return None
                benchmarks = [str(x) for x in picked]
            if not benchmarks:
                total_rows = sum(n for _, n in benches)
                console.print(f"[dim]benchmarks:[/dim] all ({total_rows:,} rows)")
                if total_rows > 10_000:
                    console.print(
                        f"[yellow]Large corpus ({total_rows:,} rows).[/yellow] "
                        f"Evaluate will subsample "
                        f"(default_eval_limit in the active AutoFA config) — "
                        "loading the full parquet into memory is unsafe."
                    )
                _warn_sensitive_benchmarks([b for b, _ in benches])
            elif len(benchmarks) == 1:
                console.print(f"[dim]benchmark:[/dim] {benchmarks[0]}")
                _warn_sensitive_benchmarks(benchmarks)
            else:
                n_sel = sum(n for b, n in benches if b in set(benchmarks))
                console.print(
                    f"[dim]benchmarks:[/dim] {len(benchmarks)} selected "
                    f"({n_sel:,} rows) — "
                    + ", ".join(benchmarks[:5])
                    + ("…" if len(benchmarks) > 5 else "")
                )
                if n_sel > 10_000:
                    console.print(
                        f"[yellow]Large selection ({n_sel:,} rows).[/yellow] "
                        "Evaluate will use a capped subsample for safety."
                    )
                _warn_sensitive_benchmarks(benchmarks)
    else:
        benchmarks = []

    return DatasetSelection(
        parquet_paths=selected,
        model_id=model_id,
        benchmarks=benchmarks,
        source_paths=source_paths,
        source_kind=source_kind,
        source_harness=str(source_harness or "unknown"),
        reference_data=list(reference_data or []),
        selection_id=selection_id,
    )


def _finish_prepared_selection(
    prepared: PreparedData,
    *,
    source_paths: list[str] | None = None,
    source_kind: str | None = None,
) -> DatasetSelection | None:
    return _finish_selection(
        prepared.parquet_paths,
        source_paths=prepared.source_paths if source_paths is None else source_paths,
        source_kind=prepared.source_kind if source_kind is None else source_kind,
        source_harness=prepared.source_harness,
        reference_data=prepared.reference_data,
        selection_id=prepared.selection_id,
    )


def _local_source_harness(sources: list[EvalSource]) -> str:
    """Infer the producer harness without confusing it with AutoFA's agent."""
    harnesses: list[str] = []
    for source in sources:
        if source.kind == "evalrdl":
            from autofa.data.local_import import detect_reader

            harnesses.append(detect_reader(source.path) or "unknown")
            continue
        if source.kind == "gym_dir":
            metadata = load_gym_metadata(source.path)
            harnesses.append(str(metadata.get("harness") or "nemo_gym"))
            continue
        if source.kind == "jsonl":
            metadata = load_gym_metadata(source.path.parent)
            harnesses.append(
                str(metadata.get("harness") or ("nemo_gym" if metadata else "unknown"))
            )
            continue
        harnesses.append("unknown")
    return summarize_source_harness(harnesses)


def _pick_sources(
    root: Path, sources: list[EvalSource]
) -> tuple[str, list[EvalSource]] | None:
    """Choose which discovered sources to use."""
    # Single gym/jsonl unit
    if len(sources) == 1 and sources[0].kind in {"gym_dir", "jsonl", "evalrdl"}:
        s = sources[0]
        console.print(f"[dim]Source:[/dim] {s.hint or s.path}")
        return s.kind, [s]

    if root.is_file():
        return sources[0].kind, sources

    # Mixed / multiple parquets (or multiple gym children)
    console.print(f"[dim]Folder:[/dim] {root}")
    kinds = {s.kind for s in sources}
    if kinds == {"parquet"}:
        console.print(f"[dim]Found {len(sources)} parquet file(s) underneath[/dim]")
        if len(sources) > 1:
            while True:
                mode = select_option(
                    f"Use entire folder ({len(sources)} parquet files)?",
                    [
                        MenuItem(f"Yes — all {len(sources)} files", "all"),
                        MenuItem("No — pick specific files…", "pick"),
                    ],
                    allow_cancel=True,
                )
                if mode is None:
                    return None
                if mode.value == "all":
                    return "parquet", sources
                paths = [s.path for s in sources]
                picked = select_paths_multi(
                    paths, title="Toggle files with Space, Enter to confirm"
                )
                if not picked:
                    continue  # Esc from multi-pick → all/pick again
                picked_set = {Path(p).resolve() for p in picked}
                return "parquet", [s for s in sources if s.path.resolve() in picked_set]
        return "parquet", sources

    # Multiple gym dirs / jsonls
    console.print(f"[dim]Found {len(sources)} eval source(s)[/dim]")
    if len(sources) == 1:
        return sources[0].kind, sources
    while True:
        mode = select_option(
            f"Use all {len(sources)} sources?",
            [
                MenuItem(f"Yes — all {len(sources)}", "all"),
                MenuItem("No — pick one…", "pick"),
            ],
            allow_cancel=True,
        )
        if mode is None:
            return None
        if mode.value == "all":
            kind = sources[0].kind if len(kinds) == 1 else "mixed"
            return kind, sources
        choice = select_option(
            "Select source",
            [MenuItem(s.hint or str(s.path), i) for i, s in enumerate(sources)],
            allow_back=True,
        )
        if is_menu_back(choice):
            continue
        if choice is None:
            return None
        s = sources[int(choice.value)]
        return s.kind, [s]


def _materialize_sources(
    sources: list[EvalSource],
    *,
    efficiency_only: bool = False,
) -> list[Path]:
    out: list[Path] = []
    for s in sources:
        if s.kind == "parquet" and s.path.is_dir():
            out.extend(materialize_eval_source(EvalSource(kind="parquet", path=path))
                       for path in discover_parquets(s.path))
        else:
            out.append(
                materialize_eval_source(s, efficiency_only=efficiency_only)
            )
    # Dedupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


# Names that often trip Anthropic Usage Policy when the agent reads sample text.
_SENSITIVE_BENCH_HINTS = ("cbrne", "wmd", "biosecurity", "cyber-offense")


def _warn_sensitive_benchmarks(names: list[str]) -> None:
    hit = [b for b in names if any(h in b.lower() for h in _SENSITIVE_BENCH_HINTS)]
    if not hit:
        return
    console.print(
        f"[yellow]Note:[/yellow] selected benchmark(s) {', '.join(hit)} may contain "
        "policy-sensitive content. Claude can refuse mid-discover — prefer a safer "
        "subset if that happens."
    )
