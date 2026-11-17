# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved-only discovery delivery and readback, outside the direct append API.

An uncertain append is reconciled by immutable content, never blindly repeated.
The service does not offer idempotency, so absence after a timeout is not proof
that an upload cannot still commit. Such attempts remain explicitly uncertain.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rich.console import Console

from autofa.config import Config, RunConfig, load_config
from autofa.core.library import FunctionLibrary
from autofa.data.remote_library_api import (
    MAX_REPORT_BYTES,
    RemoteLibraryClient,
    RemoteLibraryError,
    client_from_config,
)
from autofa.data.remote_library_publish import publish_run, remote_publish_target
from autofa.reports.failure_cards import _extract_autofa_data
from autofa.session.recovery import run_lock
from autofa.util.atomic import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _library(path: Path) -> dict[str, Any]:
    data = FunctionLibrary.load(path).to_dict()
    data["functions"] = sorted(data.get("functions", []), key=lambda row: row["name"])
    return data


def _report_projection(data: dict[str, Any]) -> dict[str, Any]:
    # Trace links/sidecars may be hydrated by the receiving service. Scores,
    # definitions, cohort denominators and source identities must survive unchanged.
    meta = data.get("meta", {})
    references = [
        {
            key: item[key]
            for key in ("kind", "store_id", "run_id", "source_harness", "export_url")
            if key in item
        }
        for item in meta.get("reference_data", [])
    ]
    coverage = meta.get("library_quality_coverage") or {}
    coverage_keys = [
        "available",
        "coverage",
        "task_coverage",
        "minimum_grade",
        "n_good_detectors",
        "n_qualifying_detectors",
        "assessment_scope",
    ]
    # Hydration may lose diagnostic inventory details for an unavailable union.
    # A displayed numerical recall must retain its exact denominator and scope.
    if coverage.get("coverage") is not None or coverage.get("task_coverage") is not None:
        coverage_keys.extend(
            [
                "n_failed_attempts",
                "n_matched_failed_attempts",
                "n_failed_tasks",
                "n_matched_failed_tasks",
                "corpus_fingerprint",
                "cohort_fingerprint",
            ]
        )
    coverage_core = {key: coverage.get(key) for key in coverage_keys}
    return {
        "meta": {
            key: meta.get(key)
            for key in (
                "model_id",
                "benchmark",
                "n_functions",
                "n_items",
                "n_failures",
                "source_harness",
            )
        },
        "coverage": coverage_core,
        "reference_data": sorted(references, key=lambda item: _canonical(item)),
        "functions": sorted(
            (
                {
                    key: fn.get(key)
                    for key in (
                        "name",
                        "source",
                        "description",
                        "verdict",
                        "metrics",
                        "behavior_validation",
                    )
                }
                for fn in data.get("functions", [])
            ),
            key=lambda fn: fn["name"],
        ),
    }


def _metrics(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("meta", {})
    coverage = meta.get("library_quality_coverage", {})
    return {
        "model_id": meta.get("model_id"),
        "benchmark": meta.get("benchmark"),
        "n_items": meta.get("n_items"),
        "n_failures": meta.get("n_failures"),
        "coverage": {
            key: value for key, value in coverage.items() if not isinstance(value, (dict, list))
        },
        "detectors": [
            {
                "name": row.get("name"),
                "verdict": row.get("verdict"),
                "recall": row.get("metrics", {}).get("recall"),
                "task_recall": row.get("metrics", {}).get("task_recall"),
            }
            for row in data.get("functions", [])
        ],
    }


def _outcome(run: RunConfig) -> dict[str, Any]:
    value = _read(run.run_dir() / "finalization_outcome.json")
    lifecycle = _read(run.run_dir() / "supervision" / "lifecycle.json")
    if (
        lifecycle.get("action") == "stop"
        and value.get("state") not in {"complete", "completed", "completed_with_gaps"}
        and (run.run_dir() / "library.json").is_file()
    ):
        # Saved-only verification runs under the native run lock, so no owner
        # can still be writing this snapshot. A stop does not imply validation.
        value = {
            "state": "stopped",
            "report_generated": True,
            "resumable": True,
            "stop_reason": "user_stop",
            "retained_results_complete": False,
        }
    target = _read(run.run_dir() / "discovery_target.json")
    return {
        **{
            key: value.get(key)
            for key in (
                "state",
                "report_generated",
                "resumable",
                "stop_reason",
                "retained_results_complete",
            )
        },
        "target": {
            key: target.get(key)
            for key in (
                "min_failure_modes",
                "n_validated_failure_modes",
                "minimum_met",
                "validation_scope",
            )
        },
    }


def _report_path(run: RunConfig, cfg: Config) -> Path:
    from autofa.local_service import is_local_service_url

    local = is_local_service_url(cfg.remote_library_base_url) or cfg.remote_library_mode == "server"
    return run.run_dir() / ("failure_cards_local.html" if local else "failure_cards_cloud.html")


def _snapshot(run: RunConfig, cfg: Config) -> dict[str, Any]:
    report = _report_path(run, cfg)
    return {
        "library_sha256": _hash((run.run_dir() / "library.json").read_bytes()),
        "library_semantic_sha256": _hash(_canonical(_library(run.run_dir() / "library.json"))),
        "report_sha256": _hash(report.read_bytes()) if report.exists() else None,
        "report_path": str(report),
    }


def _verify_version(
    run: RunConfig,
    cfg: Config,
    client: RemoteLibraryClient,
    version: Any,
    *,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        version.source_run_id != run.run_id
        or version.branch_key != remote_publish_target(run)
        or version.model_id != run.library_model_id()
    ):
        raise ValueError("Immutable publication has a different source run, branch or model")
    snapshot = _snapshot(run, cfg)
    library_path = run.run_dir() / "library.json"
    report_path = Path(snapshot["report_path"])
    if prepared:
        library_path, report_path = Path(prepared["library_path"]), Path(prepared["report_path"])
    if not report_path.exists():
        raise ValueError("Saved publication report is missing; cannot verify its metrics")
    with TemporaryDirectory(prefix="autofa-readback-") as directory:
        downloaded = client.download_version(version, Path(directory) / "library.json")
        if _library(downloaded) != _library(library_path):
            raise ValueError("Immutable publication does not contain the current saved library")
    if not version.failure_card_url:
        raise ValueError("Immutable version has no report URL")
    remote_html = client._request("GET", version.failure_card_url, max_bytes=MAX_REPORT_BYTES)
    remote_report = _extract_autofa_data(remote_html.decode("utf-8"))
    local_report = _extract_autofa_data(report_path.read_text(encoding="utf-8"))
    remote_projection, local_projection = (
        _report_projection(remote_report),
        _report_projection(local_report),
    )
    if not local_projection["reference_data"] and run.reference_data:
        # Managed Local Library reports restore sources from the saved run registry.
        # The upload's metadata intentionally carries no cloud API references.
        local_projection["reference_data"] = _report_projection(
            {"meta": {"reference_data": run.reference_data}}
        )["reference_data"]
    if remote_projection != local_projection:
        raise ValueError("Immutable report metrics or source identities differ from saved report")
    return {
        "status": "verified",
        "run_id": run.run_id,
        "verified_at": datetime.now(UTC).isoformat(),
        "service_url": client.base_url.rstrip("/"),
        "library_id": version.library_id,
        "branch_id": version.branch_id,
        "version_id": version.id,
        "source_run_id": version.source_run_id,
        "url": client.dashboard_url(
            version.library_id, branch_id=version.branch_id, version_id=version.id
        ),
        "snapshot": snapshot,
        "remote_report_sha256": _hash(remote_html),
        "metrics": _metrics(remote_report),
        "finalization": _outcome(run),
    }


def _validate_destination(
    run: RunConfig, cfg: Config, client: RemoteLibraryClient | None = None
) -> None:
    """Honor a managed run's agreed service even after its mutable global config changes."""
    path = run.run_dir() / "setup_contract.json"
    if not path.exists():
        return
    from autofa.setup_contract import _read_contract

    # Verify the saved fingerprint's integrity, including its original runtime;
    # publication/Resume may legitimately use a newer native implementation.
    agreed = _read_contract(path)["settings"]
    fields = ("remote_library_base_url", "remote_library_mode", "remote_library_token_env")
    mismatches = [field for field in fields if agreed.get(field) != getattr(cfg, field)]
    if client is not None and client.base_url.rstrip("/") != cfg.remote_library_base_url.rstrip(
        "/"
    ):
        mismatches.append("client.base_url")
    if mismatches:
        raise ValueError(
            "Publication differs from the run's agreed destination: "
            + ", ".join(mismatches)
            + ". Use its frozen native configuration."
        )


def verify_saved_publication(
    run: RunConfig, cfg: Config, *, client: RemoteLibraryClient | None = None
) -> dict[str, Any]:
    """Read back an immutable current snapshot, or return pending/uncertain honestly."""
    _validate_destination(run, cfg)
    resolved = client or client_from_config(cfg)
    _validate_destination(run, cfg, resolved)
    attempt = _read(run.run_dir() / "delivery_attempt.json")
    if attempt and attempt.get("service_url") != resolved.base_url.rstrip("/"):
        raise ValueError("Delivery destination changed; resolve the previous attempt first")
    candidates = []
    if run.remote_library_id and run.remote_branch_id:
        # Include every listed version: the response to the newest append may be lost.
        if run.remote_version_id:
            try:
                candidates.append(
                    resolved.get_version(
                        run.remote_library_id, run.remote_branch_id, run.remote_version_id
                    )
                )
            except RemoteLibraryError as exc:
                if exc.status != 404:
                    raise
        candidates.extend(
            version
            for version in resolved.list_versions(run.remote_library_id, run.remote_branch_id)[:64]
            if version.id != run.remote_version_id
        )
    elif attempt:
        # A first upload can commit before the client learns its library/branch IDs.
        for library in resolved.list_libraries()[:64]:
            for branch in resolved.list_branches(library.id)[:64]:
                if branch.branch_key == remote_publish_target(run):
                    candidates.extend(resolved.list_versions(library.id, branch.id)[:64])
    mismatch = None
    unresolved = attempt.get("status") in {"uploading", "uncertain"}
    prepared = attempt.get("prepared") if unresolved else None
    for version in candidates:
        if version.source_run_id != run.run_id or version.branch_key != remote_publish_target(run):
            continue
        try:
            result = _verify_version(run, cfg, resolved, version, prepared=prepared)
        except ValueError as exc:
            mismatch = str(exc)
            continue
        # Recover lineage after a lost response; no new version is appended.
        run.remote_library_id, run.remote_branch_id = version.library_id, version.branch_id
        run.remote_version_id, run.remote_branch_key = version.id, version.branch_key
        run.save()
        changed_checkpoint = prepared and (
            _library(Path(prepared["library_path"])) != _library(run.run_dir() / "library.json")
            or _hash(Path(prepared["report_path"]).read_bytes())
            != result["snapshot"]["report_sha256"]
        )
        atomic_write_json(
            run.run_dir() / "delivery_attempt.json",
            {
                **{key: value for key, value in attempt.items() if key != "prepared"},
                "status": "verified",
                "service_url": resolved.base_url.rstrip("/"),
                "version_id": version.id,
                "snapshot": result["snapshot"],
            },
        )
        # Only one uncertain snapshot is retained. A reconciled upload no longer
        # needs full duplicate report/library artifacts in future checkpoints.
        for name in ("library.json", "report.html"):
            (run.run_dir() / "delivery_snapshot" / name).unlink(missing_ok=True)
        if changed_checkpoint:
            return {
                "status": "pending",
                "run_id": run.run_id,
                "retry_safe": True,
                "reason": "Previous uncertain append reconciled; current snapshot changed",
            }
        atomic_write_json(run.run_dir() / "delivery_verification.json", result)
        return result
    return {
        "status": "publication_uncertain" if unresolved else "pending",
        "run_id": run.run_id,
        "reason": mismatch or "No matching immutable publication is visible",
        "retry_safe": not unresolved,
    }


def saved_only_delivery(
    run: RunConfig,
    cfg: Config,
    *,
    verify_only: bool = False,
    client: RemoteLibraryClient | None = None,
) -> dict[str, Any]:
    """Called under the run lock. Never evaluates, judges, or generates highlights."""
    if set(run.analysis_outputs) != {"agent"}:
        raise ValueError("Saved-only delivery currently supports discovery-only runs")
    _validate_destination(run, cfg)
    resolved = client or client_from_config(cfg)
    _validate_destination(run, cfg, resolved)
    previous = verify_saved_publication(run, cfg, client=resolved)
    if verify_only or previous["status"] in {"verified", "publication_uncertain"}:
        return previous
    receipt = run.run_dir() / "delivery_attempt.json"
    attempt = {
        "version": 1,
        "status": "prepared",
        "run_id": run.run_id,
        "service_url": resolved.base_url.rstrip("/"),
        "started_at": datetime.now(UTC).isoformat(),
        "snapshot": _snapshot(run, cfg),
    }
    atomic_write_json(receipt, attempt)

    class PreparedClient:
        def __getattr__(self, name: str) -> Any:
            return getattr(resolved, name)

        def upload_library(self, **kwargs: Any) -> Any:
            artifacts = run.run_dir() / "delivery_snapshot"
            artifacts.mkdir(exist_ok=True)
            payload = kwargs["library_json"]
            if isinstance(payload, Path):
                payload = payload.read_bytes()
            if isinstance(payload, (str, bytes)):
                payload = json.loads(payload)
            library_path = artifacts / "library.json"
            report_path = artifacts / "report.html"
            atomic_write_json(library_path, payload)
            source = kwargs["failure_cards_html"]
            report_path.write_bytes(source.read_bytes() if isinstance(source, Path) else source)
            atomic_write_json(
                receipt,
                {
                    **attempt,
                    "prepared": {
                        "library_path": str(library_path),
                        "report_path": str(report_path),
                    },
                },
            )
            return RemoteLibraryClient.upload_library(self, **kwargs)

        def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
            # Preparation/packing/size checks have completed. Only this boundary
            # admits the append; a local preparation failure remains safe to retry.
            atomic_write_json(receipt, {**_read(receipt), "status": "uploading"})
            try:
                return resolved._request_json(method, path, **kwargs)
            except RemoteLibraryError as exc:
                if exc.status is not None and 400 <= exc.status < 500:
                    atomic_write_json(receipt, {**_read(receipt), "status": "rejected"})
                raise

    try:
        publish_run(
            run,
            cfg,
            trigger="manual_cli",
            client=PreparedClient(),
            console=Console(file=io.StringIO()),
            show_progress=False,
        )
        result = verify_saved_publication(run, cfg, client=resolved)
        if result["status"] != "verified":
            raise RuntimeError(result.get("reason", "Publication readback failed"))
        return result
    except BaseException as exc:
        # Ctrl+C and process termination are not confirmed absence at the server.
        current = _read(receipt)
        atomic_write_json(
            receipt,
            {
                **current,
                "status": (
                    "uncertain"
                    if current.get("status") == "uploading"
                    else current.get("status", "prepared")
                ),
                "error_type": type(exc).__name__,
            },
        )
        raise


def local_saved_delivery(run: RunConfig, cfg: Config, *, verify_only: bool) -> dict[str, Any]:
    """Render a local discovery snapshot without contacting a service or model."""
    if set(run.analysis_outputs) != {"agent"}:
        raise ValueError("Saved-only delivery currently supports discovery-only runs")
    path = run.run_dir() / "failure_cards.html"
    receipt_path = run.run_dir() / "local_delivery.json"
    saved = _read(receipt_path)
    library_hash = _hash((run.run_dir() / "library.json").read_bytes())
    if verify_only:
        if (
            not path.exists()
            or saved.get("library_sha256") != library_hash
            or saved.get("report_sha256") != _hash(path.read_bytes())
        ):
            return {
                "status": "pending",
                "run_id": run.run_id,
                "reason": "Local snapshot needs rendering",
            }
    else:
        from autofa.mcp.runtime import load_runtime_from_run
        from autofa.reports.failure_cards import write_failure_cards

        runtime = load_runtime_from_run(run, cfg)
        runtime.persist_library()
        write_failure_cards(
            runtime.library,
            out=path,
            dataset=runtime.dataset,
            source_harness=run.source_harness,
            reference_data=run.reference_data,
        )
        saved = {
            "library_sha256": _hash((run.run_dir() / "library.json").read_bytes()),
            "report_sha256": _hash(path.read_bytes()),
            "report_path": str(path),
        }
        atomic_write_json(receipt_path, saved)
    return {
        "status": "verified",
        "run_id": run.run_id,
        "url": str(path),
        "snapshot": saved,
        "metrics": _metrics(_extract_autofa_data(path.read_text())),
        "finalization": _outcome(run),
        "verified_at": datetime.now(UTC).isoformat(),
    }


def main(argv: list[str]) -> int:
    """Machine interface kept separate from interactive bootstrap and highlighting."""
    parser = argparse.ArgumentParser(prog="autofa publish")
    parser.add_argument("--run", required=True)
    parser.add_argument("--saved-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.saved_only:
        parser.error("--json/--verify-only require --saved-only")
    try:
        cfg = load_config()
        run = RunConfig.load(args.run)
        with run_lock(run.run_dir()):
            if args.local_only:
                result = local_saved_delivery(run, cfg, verify_only=args.verify_only)
            else:
                result = saved_only_delivery(run, cfg, verify_only=args.verify_only)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "verified" else 2
    except (Exception, KeyboardInterrupt) as exc:
        from autofa.util.model_errors import safe_error_detail

        print(
            json.dumps({"status": "blocked", "run_id": args.run, "reason": safe_error_detail(exc)})
        )
        return 2
