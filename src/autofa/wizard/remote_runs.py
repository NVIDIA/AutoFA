# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive EvalRDL remote server run picker for the dataset wizard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.status import Status

from autofa.config import load_config
from autofa.data.reference_data import canonical_export_url, remote_reference_details
from autofa.data.remote_api import (
    DEFAULT_DATA_API_BASE_URL,
    DataApiClient,
    DataApiError,
    current_remote_export,
    materialize_remote_run,
)
from autofa.data.source_export import (
    materialize_efficiency_tables,
    materialize_source_traces,
)
from autofa.ui.errors import set_action_stage
from autofa.ui.menus import prepare_step
from autofa.ui.rollout_table import select_rollout_runs
from autofa.ui.theme import GREEN

console = Console()


class _TraceImportProgress:
    """Render transfers inside the existing spinner; reset speed/ETA per file."""

    def __init__(self, status: Status):
        self.status = status
        self.progress = Progress(
            TextColumn("{task.description}"), BarColumn(bar_width=None), TaskProgressColumn(),
            DownloadColumn(binary_units=True), TransferSpeedColumn(),
            TextColumn("ETA"), TimeRemainingColumn(),
            console=console, auto_refresh=False, expand=False,
        )
        self.task = None
        self.label = ""

    def _download(self, label: str, done: int, total: int | None) -> None:
        if self.task is None or self.label != label or done == 0:
            if self.task is not None:
                self.progress.remove_task(self.task)
            self.task = self.progress.add_task(label, total=total, completed=done)
        else:
            self.progress.update(self.task, total=total, completed=done)
        self.label = label
        self.status.update(self.progress)

    def trace_download(self, done: int, total: int | None) -> None:
        self._download("Downloading trace tables", done, total)

    def raw_download(self, done: int, total: int) -> None:
        self._download("Downloading raw log", done, total)

    def activity(self, message: str) -> None:
        self.label = ""
        self.status.update(message)

    def prepared(self, done: int, total: int) -> None:
        self.activity(f"Preparing rollout traces: {done}/{total}")


@dataclass(frozen=True)
class RemoteRunDownload:
    """One materialized Data API run and its durable provenance."""

    parquet_path: Path
    export_url: str
    run_id: str
    source_harness: str = "unknown"
    evidence_scope: str = ""

    def reference_data(self) -> dict[str, str]:
        result = {
            "run_id": self.run_id,
            "export_url": self.export_url,
            "source_harness": self.source_harness,
            "parquet_path": str(self.parquet_path.resolve()),
        }
        if self.evidence_scope:
            result["evidence_scope"] = self.evidence_scope
        return result


def pick_and_download_remote_run(
    *,
    base_url: str = DEFAULT_DATA_API_BASE_URL,
    run_id: str | None = None,
    force: bool = False,
    efficiency_only: bool = False,
) -> list[RemoteRunDownload] | None:
    """List remote runs (or use ``run_id``), download AutoFA parquet(s).

    Interactive mode downloads every checked row after the table is submitted.
    Returns one record per selected run, or ``None`` if the user backs out of
    the remote picker to the previous wizard step.
    """
    api = DataApiClient(base_url)

    if run_id:
        rid = run_id.strip()
        set_action_stage("look up remote run", api_url=api.base_url, source_run_id=rid)
        with Status(
            f"[{GREEN}]Fetching run {rid}…[/{GREEN}]",
            console=console,
            spinner="dots",
            spinner_style=GREEN,
        ):
            try:
                catalog = api.get_run(rid)
            except DataApiError as e:
                raise SystemExit(f"Remote run lookup failed: {e}") from e
        return [
            _download_one(
                api,
                rid,
                catalog,
                force=force,
                efficiency_only=efficiency_only,
            )
        ]

    return _interactive_pick_and_download(
        api,
        force=force,
        efficiency_only=efficiency_only,
    )


def _download_one(
    api: DataApiClient,
    rid: str,
    catalog: dict | None,
    *, force: bool = False,
    efficiency_only: bool = False,
) -> RemoteRunDownload:
    canonical_id = (catalog or {}).get("run_id")
    if isinstance(canonical_id, str) and canonical_id.strip():
        rid = canonical_id.strip()
    set_action_stage("download AutoFA export", api_url=api.base_url, source_run_id=rid)
    source_url = canonical_export_url(api.base_url, rid)
    already_cached = not force and current_remote_export(rid, base_url=api.base_url) is not None

    try:
        if already_cached:
            with Status(
                f"[{GREEN}]Using cached AutoFA export ({rid})…[/{GREEN}]",
                console=console,
                spinner="dots",
                spinner_style=GREEN,
            ):
                parquet = materialize_remote_run(
                    rid,
                    base_url=api.base_url,
                    client=api,
                    catalog=catalog,
                )
        else:
            bar_width = max(12, console.width // 2)
            with Progress(
                SpinnerColumn(style=GREEN),
                TextColumn(f"[bold {GREEN}]{{task.description}}[/]"),
                BarColumn(
                    bar_width=bar_width,
                    style=f"dim {GREEN}",
                    complete_style=GREEN,
                    finished_style=f"bold {GREEN}",
                    pulse_style=GREEN,
                ),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
                expand=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Downloading AutoFA export ({rid})",
                    total=None,
                )

                def on_progress(done: int, total: int | None) -> None:
                    task = progress.tasks[task_id]
                    if total is not None and task.total != total:
                        progress.update(task_id, total=total)
                    progress.update(task_id, completed=done)

                parquet = materialize_remote_run(
                    rid,
                    base_url=api.base_url,
                    client=api,
                    catalog=catalog,
                    progress=on_progress,
                    force=force,
                )
    except DataApiError as e:
        raise SystemExit(f"Download failed: {e}") from e

    full_traces = load_config().remote_full_traces and not efficiency_only
    if efficiency_only:
        set_action_stage("retrieve efficiency tables", source_run_id=rid)
        try:
            with Status("Downloading efficiency tables...", console=console) as status:
                display = _TraceImportProgress(status)
                parquet = materialize_efficiency_tables(
                    parquet,
                    api=api,
                    run_id=rid,
                    catalog=catalog,
                    download_progress=display.trace_download,
                    activity=display.activity,
                )
        except (DataApiError, OSError, ValueError) as exc:
            raise SystemExit(f"Efficiency table retrieval failed: {exc}") from exc
    elif full_traces:
        set_action_stage("retrieve original rollout traces", source_run_id=rid)
        try:
            with Status(
                "Downloading bulk rollout traces...", console=console,
            ) as status:
                display = _TraceImportProgress(status)
                parquet = materialize_source_traces(
                    parquet, api=api, run_id=rid, catalog=catalog,
                    progress=display.prepared, download_progress=display.trace_download,
                    raw_download_progress=display.raw_download, activity=display.activity,
                )
        except (DataApiError, OSError, ValueError) as exc:
            raise SystemExit(
                f"Full source retrieval failed: {exc}. No analysis was started. "
                "Fix/retry source access; remote_full_traces=false explicitly selects "
                "the limited summary export, which cannot validate harness behavior."
            ) from exc
    else:
        console.print(
            "[yellow]Summary-only API export: intermediate messages, tool observations, "
            "images and artifact contents may be absent. "
            "Harness inspection is not validated by this input.[/yellow]"
        )
    source_harness, evidence_scope, manifest = remote_reference_details(
        parquet,
        catalog,
        full_traces=full_traces,
        efficiency_only=efficiency_only,
    )
    if manifest is not None:
        console.print(
            f"[dim]Original source:[/dim] {manifest['n_rows']} rollouts, "
            f"{manifest['n_events']} events, {manifest['n_tool_calls']} tool calls. "
            f"{manifest.get('n_initial_requests_recovered', 0)} initial requests recovered. "
            f"{manifest.get('n_model_calls_recovered', 0)} raw model calls; "
            f"{manifest.get('n_raw_trajectories_recovered', 0)} missing trajectories recovered. "
            "Images and artifact files are not inspected."
        )
        for reason, count in manifest["limitations"].items():
            console.print(
                f"[yellow]Evidence limitation:[/yellow] {reason}: {count} occurrences",
                markup=True,
            )
        for reason, count in manifest.get("notices", {}).items():
            console.print(f"[dim]Source note:[/dim] {reason}: {count} model requests", markup=True)
    console.print(f"[dim]Cached:[/dim] {parquet}")
    if catalog:
        model = catalog.get("model") or "?"
        bench = catalog.get("benchmark") or "?"
        n = catalog.get("n_rollouts")
        n_s = f"{n:,}" if isinstance(n, int) else "?"
        console.print(
            f"[dim]EvalRDL run:[/dim] {model} · {bench} · {n_s} rollouts · {rid}"
        )
    return RemoteRunDownload(
        parquet_path=parquet,
        export_url=source_url,
        run_id=rid,
        source_harness=source_harness or "unknown",
        evidence_scope=evidence_scope,
    )


def pick_remote_runs(api: DataApiClient) -> list[dict] | None:
    """Select catalog rows without downloading them."""
    prepare_step()
    console.print(
        Panel(
            f"[bold {GREEN}]EvalRDL remote server[/]\n"
            f"[dim]{api.base_url}[/dim]\n"
            f"[dim]Search and select rollout sets from the complete catalog.[/dim]",
            border_style=GREEN,
        )
    )

    with Status(
        f"[{GREEN}]Loading rollout catalog…[/{GREEN}]",
        console=console,
        spinner="dots",
        spinner_style=GREEN,
    ):
        set_action_stage("load remote rollout catalog", api_url=api.base_url)
        try:
            runs = api.list_runs()
        except DataApiError as e:
            raise SystemExit(f"Failed to list runs: {e}") from e

    if not runs:
        raise SystemExit("Remote catalog contains no rollout sets")
    selected = select_rollout_runs(runs)
    if selected is None:
        return None
    if not selected:
        raise SystemExit("Catalog returned rollout sets without run_id")
    return selected


def _interactive_pick_and_download(
    api: DataApiClient,
    *, force: bool = False,
    efficiency_only: bool = False,
) -> list[RemoteRunDownload] | None:
    selected = pick_remote_runs(api)
    if selected is None:
        return None

    results = [
        _download_one(
            api,
            str(catalog["run_id"]),
            catalog,
            force=force,
            efficiency_only=efficiency_only,
        )
        for catalog in selected
    ]

    if len(results) > 1:
        console.print(
            f"[dim]Selected {len(results)} remote runs — "
            f"AutoFA will use them together.[/dim]"
        )
    return results
