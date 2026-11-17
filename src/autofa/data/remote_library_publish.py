# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared strict and best-effort publication of saved AutoFA results."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.data.dataset_identity import normalize_data_identity, publication_data_identity
from autofa.data.evalrdl_efficiency import validate_efficiency_run_selection
from autofa.data.reference_data import (
    data_api_location,
    local_viewer_base,
    normalize_reference_data,
    public_reference_data,
)
from autofa.data.remote_library_api import (
    RemoteDataReference,
    RemoteLibraryClient,
    RemoteLibraryError,
    RemotePublication,
    client_from_config,
    library_service_name,
)
from autofa.util.atomic import atomic_write_json

if TYPE_CHECKING:
    from autofa.mcp.runtime import Runtime

type PublishTrigger = Literal[
    "manual_cli",
    "browse_menu",
    "session_end",
]
VALID_PUBLISH_TRIGGERS = frozenset({"manual_cli", "browse_menu", "session_end"})


class _PublicationTargetError(RemoteLibraryError):
    """A run's known publication belongs to a different service."""


def remote_publish_target(run: RunConfig) -> str:
    """Return the stable v2 branch key for publishing this run."""
    try:
        return run.publication_branch_key()
    except ValueError as exc:
        raise RemoteLibraryError(str(exc)) from exc


def _validated_publish_trigger(trigger: PublishTrigger | str) -> PublishTrigger:
    value = str(trigger).strip()
    if value not in VALID_PUBLISH_TRIGGERS:
        allowed = ", ".join(sorted(VALID_PUBLISH_TRIGGERS))
        raise ValueError(f"Invalid publish trigger {value!r}; expected one of: {allowed}")
    return cast("PublishTrigger", value)


def publish_run(
    run: RunConfig,
    cfg: Config,
    *,
    trigger: PublishTrigger,
    runtime: Runtime | Any | None = None,
    client: RemoteLibraryClient | Any | None = None,
    console: Console | Any | None = None,
    show_progress: bool = True,
) -> RemotePublication:
    """Publish available results without evaluating detectors or retrying reviews."""
    publish_trigger = _validated_publish_trigger(trigger)
    if "efficiency" in set(run.analysis_outputs or []):
        validate_efficiency_run_selection(run)
    service_url = _publication_origin(cfg.remote_library_base_url)
    previous_service = _previous_publication_service(run)
    if (
        (run.remote_library_id or run.remote_branch_id)
        and previous_service and service_url and previous_service != service_url
    ):
        raise _PublicationTargetError(
            f"This run's library was published to {previous_service}, but the configured "
            f"library service is {service_url}. Start a new run for this service or "
            "restore the previous library URL before publishing this run."
        )
    data_identity = publication_data_identity(run.parquet_paths, reference_data=run.reference_data)
    _publication_status(run, publish_trigger, "publishing", service_url=service_url)
    try:
        remote = _publish_run(run, cfg, trigger=publish_trigger, runtime=runtime,
                              client=client, console=console, show_progress=show_progress,
                              data_identity=data_identity)
    except Exception as exc:
        _publication_status(run, publish_trigger, "failed", error=str(exc))
        raise
    _publication_status(run, publish_trigger, "published",
                        url=remote.dashboard_url or remote.failure_card_url,
                        version_id=remote.version_id,
                        library_sha256=_library_fingerprint(run), data_identity=data_identity)
    return remote


def _library_fingerprint(run: RunConfig) -> str:
    try:
        return hashlib.sha256((run.run_dir() / "library.json").read_bytes()).hexdigest()
    except OSError:
        return ""


def published_library_is_current(run: RunConfig, cfg: Config) -> bool:
    """Whether the last successful publication already contains the saved library."""
    try:
        status = json.loads((run.run_dir() / "publication.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(status, dict) or status.get("state") != "published":
        return False
    try:
        saved_identity = normalize_data_identity(status.get("data_identity"))
    except ValueError:
        return False
    fingerprint = status.get("library_sha256")
    return (
        bool(fingerprint)
        and fingerprint == _library_fingerprint(run)
        and _previous_publication_service(run) == _publication_origin(cfg.remote_library_base_url)
        and saved_identity == publication_data_identity(
            run.parquet_paths, reference_data=run.reference_data,
        )
    )


def _publication_status(run: RunConfig, trigger: str, state: str, **details) -> None:
    try:
        previous_service = _previous_publication_service(run)
        atomic_write_json(run.run_dir() / "publication.json", {
            "state": state, "trigger": trigger, "run_id": run.run_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "retry_command": f"autofa publish --run {run.run_id}",
            **({"service_url": previous_service} if previous_service else {}),
            **details,
        })
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not save library publication status", exc_info=True,
        )


def _publication_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        default_port = 80 if parsed.scheme == "http" else 443
        netloc = (
            f"{hostname}:{parsed.port}" if parsed.port not in {None, default_port} else hostname
        )
        return urlunsplit((parsed.scheme, netloc, "", "", ""))
    except ValueError:
        return ""


def _previous_publication_service(run: RunConfig) -> str:
    try:
        status = json.loads((run.run_dir() / "publication.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(status, dict):
        return ""
    return _publication_origin(str(status.get("service_url") or ""))


def _publish_run(
    run: RunConfig,
    cfg: Config,
    *,
    trigger: PublishTrigger,
    data_identity: dict[str, Any] | None,
    runtime: Runtime | Any | None = None,
    client: RemoteLibraryClient | Any | None = None,
    console: Console | Any | None = None,
    show_progress: bool = True,
) -> RemotePublication:
    from autofa.local_service import is_local_service_url
    from autofa.mcp.runtime import load_runtime_from_run
    from autofa.reports.failure_cards import (
        build_report_assets_zip,
        write_failure_cards,
        write_failure_cards_with_status,
    )

    con = console or Console()
    publish_trigger = _validated_publish_trigger(trigger)
    portable_backend = getattr(cfg, "remote_library_mode", "cloud") == "server"
    local_backend = is_local_service_url(cfg.remote_library_base_url) and not portable_backend
    run_references = normalize_reference_data(getattr(run, "reference_data", []))
    # The Local Library can reconstruct managed traces from its own run/store
    # registry. Send only the report; it resolves each full example on demand.
    managed_local = bool(run_references) and all(local_viewer_base(ref) for ref in run_references)
    source_harness = (
        str(getattr(run, "source_harness", None) or "unknown").strip()
        or "unknown"
    )
    analysis_outputs = set(getattr(run, "analysis_outputs", None) or [])
    efficiency_only = analysis_outputs == {"efficiency"}
    # Cloud metadata is strict; richer visibility details stay in the HTML report.
    reference_data = [
        RemoteDataReference.from_dict(reference).to_dict()
        for reference in (
            [] if local_backend or portable_backend
            else public_reference_data(getattr(run, "reference_data", []))
        )
        if data_api_location(reference) is not None
    ]
    resolved_client = client or client_from_config(cfg)
    resolved_runtime = runtime or load_runtime_from_run(run, cfg)
    # Publication is a snapshot; evaluation and explicit repair belong to discovery.
    library_payload = resolved_runtime.persist_library()
    if efficiency_only:
        from autofa.core.library import FunctionLibrary

        library_payload = FunctionLibrary(
            model_id=run.library_model_id(), benchmark=run.benchmark_label(),
        ).to_dict()
    trace_references = None
    if portable_backend and not efficiency_only:
        from pathlib import Path

        from autofa.data.local_trace_store import resolve_store
        from autofa.data.trace_upload import server_capabilities, upload_trace_stores

        capabilities = server_capabilities(resolved_client)
        managed_references = [ref for ref in run_references if local_viewer_base(ref)]
        if managed_references:
            expected_paths = {resolve_store(ref["store_id"]) / "dataset.parquet"
                              for ref in managed_references}
            actual_paths = {Path(path).expanduser().resolve()
                            for path in resolved_runtime.dataset.parquet_paths}
            if not expected_paths <= actual_paths:
                raise RemoteLibraryError(
                    "The saved trace references do not cover the run's dataset"
                )
            managed_local = expected_paths == actual_paths
            if show_progress:
                con.print("Uploading complete trace snapshots to the AutoFA server…")
            transferred = upload_trace_stores(
                resolved_client, managed_references, capabilities=capabilities,
            )
            # Mixed inputs retain their full ordinary report sidecars. Their
            # native managed-source links still resolve on the receiving server.
            if managed_local:
                trace_references = transferred
    if efficiency_only:
        from autofa.reports.efficiency_card import resolve_pricing, write_efficiency_card

        if show_progress:
            con.print("Generating efficiency report…")
        report_path, _, _ = write_efficiency_card(
            run,
            run.run_dir(),
            source_harness=source_harness,
            reference_data=reference_data,
            pricing=resolve_pricing(cfg, getattr(run, "model_id", "")),
        )
    else:
        report_writer = write_failure_cards_with_status if show_progress else write_failure_cards
        report_kwargs = {
            "out": run.run_dir() / (
                "failure_cards_local.html" if local_backend or portable_backend
                else "failure_cards_cloud.html"
            ),
            "dataset": resolved_runtime.dataset,
            "source_harness": source_harness,
            "reference_data": [
                ref for ref in run_references
                if local_backend or portable_backend or not local_viewer_base(ref)
            ],
            "prefer_source_links": not portable_backend or managed_local,
            "source_links_only": (not local_backend and not portable_backend) or managed_local,
        }
        if show_progress:
            report_kwargs["console"] = con
        report_path = report_writer(resolved_runtime.library, **report_kwargs)
        try:
            from autofa.reports.efficiency_card import attach_saved_efficiency

            attach_saved_efficiency(report_path, resolved_runtime.library)
        except Exception as exc:  # noqa: BLE001
            con.print(f"[yellow]Efficiency metrics were not added to the report: {exc}[/yellow]")
    report_assets_zip = build_report_assets_zip(
        report_path.parent,
        html_name=report_path.name,
    )
    branch_key = remote_publish_target(run)
    metadata: dict[str, Any] = {
        "branch_key": branch_key,
        "source_run_id": run.run_id,
        "model_id": run.library_model_id(),
        "harness": str(run.harness or "unknown").strip() or "unknown",
        "source_harness": source_harness,
        "reference_data": reference_data,
        # Explicit null signals incomplete source binding. Servers must not infer
        # a complete identity from only the available subset of reference_data.
        "data_identity": data_identity,
        "publish_trigger": publish_trigger,
        "benchmarks": run.resolved_benchmarks(),
    }
    if run.remote_library_id:
        metadata["library_id"] = run.remote_library_id
    if run.remote_branch_id:
        metadata["branch_id"] = run.remote_branch_id
    transfer_kwargs = {"trace_references": trace_references} if trace_references is not None else {}
    remote = resolved_client.upload_library(
        metadata=metadata,
        library_json=library_payload,
        failure_cards_html=report_path,
        report_assets_zip=report_assets_zip,
        **transfer_kwargs,
    )
    run.remote_library_id = remote.library_id
    run.remote_branch_id = remote.branch_id
    run.remote_branch_key = remote.branch_key
    run.remote_version_id = remote.version_id
    run.save()
    return remote


def maybe_publish_run(
    run: RunConfig,
    cfg: Config,
    *,
    trigger: PublishTrigger,
    runtime: Runtime | Any | None = None,
    console: Console | Any | None = None,
) -> RemotePublication | None:
    """Publish when a credential exists without risking local completion."""
    from autofa.services import remote_library_enabled

    publish_trigger = _validated_publish_trigger(trigger)
    if not remote_library_enabled(cfg):
        return None
    con = console or Console()
    service_name = library_service_name(cfg.remote_library_base_url,
                                        mode=getattr(cfg, "remote_library_mode", "cloud"))
    publication_name = ("Server" if getattr(cfg, "remote_library_mode", "cloud") == "server"
                        else "Local" if service_name == "AutoFA Local Library" else "Cloud")
    try:
        resolved_client = client_from_config(cfg)
    except Exception as exc:  # noqa: BLE001
        _publication_status(run, publish_trigger, "skipped", error=str(exc))
        con.print(
            f"[yellow]{publication_name} publication skipped; local library is safe:[/yellow] "
            f"{exc}"
        )
        return None

    try:
        remote = publish_run(
            run,
            cfg,
            trigger=publish_trigger,
            runtime=runtime,
            client=resolved_client,
            console=con,
        )
    except Exception as exc:  # noqa: BLE001
        from autofa.ui.errors import show_action_error

        if not isinstance(exc, _PublicationTargetError):
            _publication_status(run, publish_trigger, "failed", error=str(exc))
        con.print(
            "[yellow]Remote publish failed; local library is safe:[/yellow] "
            f"{exc}"
        )
        show_action_error(exc, console=con, wait=False,
                          context=f"{service_name} publication for {run.run_id}")
        con.print(f"Retry: autofa publish --run {run.run_id}", markup=False)
        return None

    base_url = cfg.remote_library_base_url.rstrip("/")
    url = (remote.dashboard_url or remote.failure_card_url
           or f"{base_url}/libraries/{remote.library_id}")
    con.print(f"[green]Published to {service_name}:[/green] {url}")
    return remote
