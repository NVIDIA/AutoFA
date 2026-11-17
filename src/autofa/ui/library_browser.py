# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive library browser (model × discovery-benchmark catalogs)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from autofa.config import Config, RunConfig, library_root, load_config
from autofa.core.library import FunctionLibrary
from autofa.data.library_provenance import (
    CatalogProvenance,
    read_catalog_provenance,
    write_catalog_provenance,
)
from autofa.data.remote_library_api import (
    RemoteBranch,
    RemoteLibrary,
    RemoteLibraryClient,
    RemoteLibraryError,
    RemoteLibraryRef,
    RemotePublication,
    RemoteVersion,
    client_from_config,
    library_service_name,
    materialize_remote_library,
)
from autofa.services import remote_library_enabled
from autofa.ui.errors import show_action_error, show_action_notice
from autofa.ui.menus import (
    MENU_BACK,
    MenuBack,
    MenuItem,
    is_menu_back,
    select_option,
    select_yes_no,
)
from autofa.ui.theme import GREEN

console = Console()


@dataclass(frozen=True)
class CatalogEntry:
    """One durable catalog: model + discovery-benchmark set."""

    model_id: str
    benchmark: str  # human label from library.json when present
    path: Path  # …/library.json
    n_functions: int = 0

    @property
    def catalog_dir(self) -> Path:
        return self.path.parent

    @property
    def model_dir(self) -> Path:
        return self.catalog_dir.parent

    def menu_label(self) -> str:
        return f"{self.model_id}  ·  {format_benchmark_label(self.benchmark)}"

    def menu_detail(self) -> str:
        n = self.n_functions
        return f"({n} fn)" if n else "(empty)"


def format_benchmark_label(raw: str) -> str:
    """Readable discovery-benchmark label for menus / titles."""
    s = (raw or "").strip() or "all"
    if s.lower() in {"all", "all-benchmarks", "all_benchmarks"}:
        return "all benchmarks"
    # Stored join uses '+'; filesystem segment may use '_'
    if "+" in s:
        parts = [p.strip() for p in s.split("+") if p.strip()]
        return " + ".join(parts) if parts else s
    return s


def list_catalog_entries(root: Path) -> list[CatalogEntry]:
    """Scan ``root/{model}/{benchmark}/library.json`` into flat catalog entries."""
    root = Path(root)
    if not root.exists():
        return []
    entries: list[CatalogEntry] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            lib_path = bench_dir / "library.json"
            if not lib_path.is_file():
                continue
            model_id, benchmark, n_fns = _read_catalog_meta(
                lib_path, model_dir.name, bench_dir.name
            )
            entries.append(
                CatalogEntry(
                    model_id=model_id,
                    benchmark=benchmark,
                    path=lib_path,
                    n_functions=n_fns,
                )
            )
    return entries


def _read_catalog_meta(
    lib_path: Path, fallback_model: str, fallback_bench: str
) -> tuple[str, str, int]:
    """Load model/benchmark/n_fns without requiring a full FunctionLibrary parse failure."""
    try:
        data = json.loads(lib_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback_model, _bench_from_dirname(fallback_bench), 0
    model = str(data.get("model_id") or "").strip() or fallback_model
    bench = str(data.get("benchmark") or "").strip() or _bench_from_dirname(fallback_bench)
    fns = data.get("functions") or []
    n = len(fns) if isinstance(fns, list) else 0
    return model, bench, n


def _bench_from_dirname(name: str) -> str:
    """Best-effort restore of '+' joined benches from a safe path segment."""
    if not name or name == "all":
        return name or "all"
    # Common pattern: swebench-pro_swebench-verified_swebench-multilingual
    if "_swebench" in name or name.count("_") >= 2:
        # Heuristic: underscores between known multi-bench dumps → treat as '+'
        # Prefer leaving as-is when library.json has the real label (usual path).
        return name.replace("_", "+")
    return name


def browse_library(root: Path | None = None) -> None:
    """Browse local catalogs, or choose local/remote from the home flow."""
    if root is not None:
        browse_local_library(root)
        return

    cfg = load_config()
    while True:
        source = select_option(
            "Browse library",
            [
                MenuItem(
                    "Local library",
                    "local",
                    detail=str(cfg.resolved_library_root()),
                ),
                *([MenuItem(library_service_name(cfg.remote_library_base_url,
                                                mode=cfg.remote_library_mode), "remote")]
                  if remote_library_enabled(cfg) else []),
            ],
            allow_cancel=True,
        )
        if source is None:
            return
        if source.value == "local":
            browse_local_library(cfg.resolved_library_root() or library_root())
            continue
        try:
            browse_remote_library(client_from_config(cfg))
        except RemoteLibraryError as exc:
            show_action_error(exc, console=console, context="Browse remote library")


def browse_local_library(root: Path | None = None) -> None:
    """Browse the existing on-disk model/benchmark catalog."""
    cfg = load_config()
    root = root or cfg.resolved_library_root() or library_root()
    root = Path(root)
    feedback: tuple[str, str] | None = None
    if not root.exists():
        show_action_notice(f"No library at {root}. Run discovery or select another folder.",
                           console=console)
        return

    while True:
        entries = list_catalog_entries(root)
        if not entries:
            show_action_notice(f"Library empty: {root}. Run discovery to create detectors.",
                               console=console)
            return

        footer = None
        if feedback is not None:
            style, message = feedback
            footer = [(f"class:{style}", f"  {message}\n")]
        choice = select_option(
            f"Library catalogs — {root}",
            [
                MenuItem(
                    e.menu_label(),
                    e,
                    detail=e.menu_detail(),
                )
                for e in entries
            ],
            allow_cancel=True,
            allow_delete=True,
            allow_publish=remote_library_enabled(cfg),
            footer=footer,
        )
        if choice is None:
            return

        entry = choice.value
        assert isinstance(entry, CatalogEntry)

        if choice.action == "publish":
            try:
                remote = _publish_catalog(entry, cfg)
                url = (
                    remote.dashboard_url
                    or remote.failure_card_url
                    or f"{cfg.remote_library_base_url.rstrip('/')}/libraries/{remote.library_id}"
                )
                service = library_service_name(cfg.remote_library_base_url,
                                               mode=cfg.remote_library_mode)
                feedback = (
                    "good",
                    f"✓ Library published in {service}"
                    f" · {url}",
                )
            except Exception as exc:  # noqa: BLE001
                feedback = ("bad", f"✗ Library publish failed · {exc}")
                show_action_error(exc, console=console, context="Publish library")
            continue

        if choice.action == "delete":
            label = f"{entry.model_id} / {format_benchmark_label(entry.benchmark)}"
            if _confirm_delete(label, entry.catalog_dir):
                shutil.rmtree(entry.catalog_dir)
                console.print(f"[{GREEN}]Deleted[/] {entry.catalog_dir}")
                model_dir = entry.model_dir
                if model_dir.exists() and not any(model_dir.iterdir()):
                    shutil.rmtree(model_dir, ignore_errors=True)
                    console.print(f"[dim]Removed empty {model_dir}[/dim]")
            continue

        if not entry.path.exists():
            show_action_error(FileNotFoundError(f"Missing library: {entry.path}"),
                              console=console)
            continue
        _open_library(entry)


def _matching_run(entry: CatalogEntry) -> RunConfig | None:
    """Resolve the catalog's source run from its provenance sidecar."""
    provenance = read_catalog_provenance(entry.path)
    if provenance is not None:
        try:
            run = RunConfig.load(provenance.source_run_id)
        except (OSError, ValueError, TypeError):
            pass
        else:
            # The sidecar is the catalog snapshot's lineage source; the run's
            # config.json may have moved on since (e.g. after a later Continue).
            # Hydrate the run in memory so p-publish targets the snapshot's branch.
            run.remote_library_id = provenance.remote_library_id
            run.remote_branch_id = provenance.remote_branch_id
            run.remote_branch_key = provenance.remote_branch_key
            run.remote_version_id = provenance.remote_version_id
            run.remote_base_version_id = provenance.remote_base_version_id
            return run
    return None


def _publish_catalog(
    entry: CatalogEntry,
    cfg: Config,
) -> RemotePublication:
    """Publish a selected catalog through its matching source run."""
    console.print(f"[{GREEN}]Publishing the library…[/]")
    run = _matching_run(entry)
    if run is None:
        raise RemoteLibraryError(
            "No matching source run was found for this catalog; "
            "use `autofa publish --run <id>` to select one explicitly"
        )

    from autofa.data.remote_library_publish import publish_run

    remote = publish_run(
        run,
        cfg,
        trigger="browse_menu",
        console=console,
        show_progress=False,
    )
    write_catalog_provenance(entry.path, CatalogProvenance.from_run(run))
    return remote


def remote_library_menu_item(entry: RemoteLibrary) -> MenuItem:
    """Render a remote catalog with the same label/detail as a local one."""
    model = entry.model_id or "?"
    benchmark = format_benchmark_label("+".join(entry.benchmarks) or "all")
    count = f"({entry.function_count} fn)" if entry.function_count else "(empty)"
    return MenuItem(
        f"{model}  ·  {benchmark}",
        entry,
        detail=f"{count} · {entry.branch_count} branch"
        f"{'es' if entry.branch_count != 1 else ''} · {entry.version_count} version"
        f"{'s' if entry.version_count != 1 else ''}",
    )


def select_remote_library(
    client: RemoteLibraryClient,
    *,
    title: str = "Remote libraries",
) -> RemoteLibraryRef | None:
    """Navigate logical library → branch → immutable version."""
    while True:
        entries = client.list_libraries()
        if not entries:
            show_action_notice("No remote libraries found for this user. Publish a local library "
                               "or select another account.", console=console)
            return None
        choice = select_option(
            title,
            [remote_library_menu_item(entry) for entry in entries],
            allow_cancel=True,
        )
        if choice is None:
            return None
        library = choice.value
        assert isinstance(library, RemoteLibrary)
        selected = _select_remote_branch_version(client, library)
        if is_menu_back(selected):
            continue
        return selected


def _select_remote_branch_version(
    client: RemoteLibraryClient,
    library: RemoteLibrary,
) -> RemoteLibraryRef | MenuBack:
    """Select a branch and concrete version, with back navigation at each level."""
    branches = sorted(
        client.list_branches(library.id),
        key=_remote_branch_sort_key,
        reverse=True,
    )
    if not branches:
        raise RemoteLibraryError(f"Remote library {library.id} has no branches")
    branch_menu_required = len(branches) != 1

    while True:
        if branch_menu_required:
            branch_choice = select_option(
                f"Select branch · {library.model_id or '?'}",
                [
                    _remote_branch_menu_item(
                        branch,
                        is_library_latest=branch.id == library.latest_branch_id,
                    )
                    for branch in branches
                ],
                allow_cancel=False,
                allow_back=True,
                default_index=0,
            )
            if is_menu_back(branch_choice):
                return MENU_BACK
            assert isinstance(branch_choice, MenuItem)
            branch = branch_choice.value
            assert isinstance(branch, RemoteBranch)
        else:
            branch = branches[0]

        versions = client.list_versions(library.id, branch.id)
        if not versions:
            raise RemoteLibraryError(f"Remote branch {branch.id} has no versions")
        version_choice = select_option(
            f"Select version · {branch.branch_key}",
            [_remote_version_menu_item(version) for version in versions],
            allow_cancel=False,
            allow_back=True,
            default_index=0,
        )
        if is_menu_back(version_choice):
            if branch_menu_required:
                continue
            return MENU_BACK
        assert isinstance(version_choice, MenuItem)
        version = version_choice.value
        assert isinstance(version, RemoteVersion)
        return RemoteLibraryRef(library=library, branch=branch, version=version)


def _remote_branch_menu_item(
    branch: RemoteBranch,
    *,
    is_library_latest: bool = False,
) -> MenuItem:
    count = f"{branch.version_count} version{'s' if branch.version_count != 1 else ''}"
    detail = [count]
    updated = _format_remote_timestamp(branch.updated_at or branch.created_at)
    if updated:
        detail.append(updated)
    if is_library_latest:
        detail.append("library latest")
    return MenuItem(branch.branch_key, branch, detail=" · ".join(detail))


def _remote_branch_sort_key(branch: RemoteBranch) -> tuple[float, str]:
    timestamp = branch.updated_at or branch.created_at
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp(), branch.id
    except (ValueError, OverflowError):
        return 0.0, branch.id


def _format_remote_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, OverflowError):
        return value


def _remote_version_menu_item(version: RemoteVersion) -> MenuItem:
    source = version.source_run_id or "unknown run"
    created = f" · {version.created_at}" if version.created_at else ""
    return MenuItem(
        f"v{version.version_number or '?'}",
        version,
        detail=f"{source}{created}",
    )


def browse_remote_library(
    client: RemoteLibraryClient,
) -> None:
    """Browse and open one exact immutable dashboard version."""
    service_name = library_service_name(
        getattr(client, "base_url", ""),
        mode="server" if getattr(client, "credential_source", "") == "server" else "cloud",
    )
    while True:
        entries = client.list_libraries()
        if not entries:
            show_action_notice(
                f"No libraries found in {service_name}. Publish a local catalog to get started.",
                console=console,
            )
            return
        choice = select_option(
            service_name,
            [remote_library_menu_item(entry) for entry in entries],
            allow_cancel=True,
            allow_delete=True,
        )
        if choice is None:
            return

        entry = choice.value
        assert isinstance(entry, RemoteLibrary)
        benchmark = "+".join(entry.benchmarks) or "all"
        label = f"{entry.model_id or '?'} / {format_benchmark_label(benchmark)}"
        if choice.action == "delete":
            console.print(f"[yellow]Delete[/] {label}")
            console.print(f"[dim]Remote library {entry.id}[/dim]")
            if select_yes_no(
                "Permanently delete this remote catalog?",
                default_yes=False,
            ):
                client.delete_library(entry.id)
                console.print(f"[{GREEN}]Deleted[/] {label}")
            continue

        selected = _select_remote_branch_version(client, entry)
        if is_menu_back(selected):
            continue
        assert isinstance(selected, RemoteLibraryRef)
        path = materialize_remote_library(selected, client=client)
        version = selected.version
        version_benchmark = "+".join(version.benchmarks)
        _open_library(
            CatalogEntry(
                model_id=version.model_id or entry.model_id,
                benchmark=version_benchmark or benchmark,
                path=path,
                n_functions=version.function_count,
            ),
            source_harness=version.source_harness,
            reference_data=[
                reference.to_dict()
                for reference in version.reference_data
            ],
        )


def _confirm_delete(label: str, path: Path) -> bool:
    console.print(f"[yellow]Delete[/] {label}")
    console.print(f"[dim]{path}[/dim]")
    return select_yes_no("Permanently delete this catalog entry?", default_yes=False) is True


def _open_library(
    entry: CatalogEntry,
    *,
    source_harness: str | None = None,
    reference_data: list[dict[str, str]] | None = None,
) -> None:
    lib = FunctionLibrary.load(entry.path)
    model = lib.model_id or entry.model_id
    bench = format_benchmark_label(lib.benchmark or entry.benchmark)

    dataset = None
    provenance = read_catalog_provenance(entry.path)
    if provenance is not None:
        source_harness = source_harness or provenance.source_harness
        reference_data = reference_data or list(provenance.reference_data)
        try:
            source_run = RunConfig.load(provenance.source_run_id)
        except (OSError, TypeError, ValueError):
            source_run = None
        if source_run is not None:
            from autofa.reports.failure_cards import try_dataset_from_dir

            dataset = try_dataset_from_dir(source_run.run_dir())
            source_harness = source_run.source_harness
            reference_data = list(source_run.reference_data)

    from autofa.ui.explorer import explore_library

    explore_library(
        lib,
        top_x=None,
        title=f"AutoFA explorer · {model} / {bench}",
        html_path=entry.path.parent / "failure_cards.html",
        dataset=dataset,
        source_harness=source_harness,
        reference_data=reference_data,
    )
