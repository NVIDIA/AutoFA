# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only discovery of saved local libraries, with immutable dashboard snapshots.

Only the dashboard database and artifact store are written. A run's configuration,
publication identity and original library/report files remain owned by AutoFA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from autofa.core.library import FunctionLibrary
from autofa.data.dataset_identity import normalize_data_identity, publication_data_identity
from autofa.data.reference_data import local_viewer_base, normalize_reference_data

from .config import Settings
from .db import Database
from .library_identity import same_library_content
from .schemas import UploadMetadata, validate_library_payload
from .storage import ArtifactStore

logger = logging.getLogger(__name__)
_REPORT_DATA = re.compile(
    rb'<script id="autofa-data" type="application/json">(.*?)</script>', re.DOTALL
)
_REPORT_NAMES = ("failure_cards_local.html", "failure_cards.html", "failure_cards_cloud.html")


class _DeferReport(Exception):
    """Prepare a large saved report after every library has been discovered."""


class _InvalidLibrary(ValueError):
    """The saved library.json cannot be indexed until that file changes."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _signature(paths: tuple[Path, ...]) -> tuple:
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            result.append((str(path), None))
    return tuple(result)


def _bounded_read(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        result = stream.read(maximum + 1)
    if len(result) > maximum:
        raise ValueError(f"local artifact exceeds size limit: {path.name}")
    return result


def _watch(watched: dict[Path, tuple], path: Path) -> None:
    watched.setdefault(path, _signature((path,))[0])


def _with_execution_status(report: bytes, execution: dict) -> bytes:
    match = _REPORT_DATA.search(report)
    if match is None or execution.get("status") == "unknown":
        return report
    data = json.loads(match.group(1))
    data.setdefault("meta", {})["agent_execution"] = execution
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").encode()
    return report[:match.start(1)] + encoded + report[match.end(1):]


def _publication_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat(timespec="microseconds")


class LocalCatalog:
    # Coalesce reads during a busy profile into at most one follow-up scan.
    refresh_interval = 2.0

    def __init__(
        self,
        settings: Settings,
        database: Database,
        store: Callable[[], ArtifactStore],
        publish: Callable[..., dict],
        write_lock: threading.RLock,
    ):
        self.settings = settings
        self.database = database
        self.store = store
        self.publish = publish
        self.write_lock = write_lock
        self._scan_lock = threading.RLock()
        self._cache: dict[str, tuple[tuple[Path, ...], tuple, str]] = {}
        self._activity_cache: dict[str, tuple] = {}
        self._invalid: dict[str, tuple] = {}
        self._stopped = threading.Event()
        self._catalog_ready = threading.Event()
        self._condition = threading.Condition()
        self._requested: threading.Event | None = None
        self._active = False
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        # HTTP readiness and dashboard reads never wait for filesystem discovery.
        with self._condition:
            if self._worker is not None or self._stopped.is_set():
                return
            self._requested = threading.Event()
            self._worker = threading.Thread(
                target=self._work, name="autofa-library-catalog", daemon=True
            )
            self._worker.start()

    def request_refresh(self) -> threading.Event | None:
        """Queue discovery without doing I/O; repeated requests share one queued pass.

        The returned event is set after that pass (including report preparation)
        completes. HTTP callers deliberately do not wait for it.
        """
        with self._condition:
            if self._stopped.is_set():
                return None
            if self._requested is None:
                self._requested = threading.Event()
            self._condition.notify()
            return self._requested

    @property
    def indexing(self) -> bool:
        with self._condition:
            return not self._stopped.is_set() and (
                self._active or self._requested is not None or not self._catalog_ready.is_set()
            )

    def request_stop(self) -> None:
        """Stop accepting work immediately, without waiting on indexing or file I/O."""
        self._stopped.set()
        with self._condition:
            self._requested = None
            self._condition.notify_all()

    def close(self) -> None:
        self.request_stop()
        if self._worker is not None:
            self._worker.join(timeout=1)

    def _work(self) -> None:
        next_refresh = 0.0
        while True:
            with self._condition:
                while not self._stopped.is_set():
                    delay = next_refresh - time.monotonic()
                    if self._requested is not None and delay <= 0:
                        break
                    self._condition.wait(timeout=max(0, delay) if self._requested else None)
                if self._stopped.is_set():
                    return
                completed = self._requested
                self._requested = None
                self._active = True
            try:
                self._refresh()
                self._catalog_ready.set()
                # Finish enrichment before handling queued reads so a busy browser
                # cannot continually restart discovery and starve report preparation.
                if not self._stopped.is_set():
                    self._enrich_pending()
            except Exception:
                logger.warning("Unable to refresh saved libraries", exc_info=True)
            finally:
                with self._condition:
                    self._active = False
                    next_refresh = time.monotonic() + self.refresh_interval
                    completed.set()

    def _refresh(self) -> None:
        """Worker-only scan; unchanged files require only stat()."""
        from .lazy_report import renderer_files

        with self._scan_lock:
            with self.database.read() as connection:
                indexed = {
                    row["source_path"]: dict(row)
                    for row in connection.execute(
                        "SELECT snapshot.*, version.source_run_id "
                        "FROM local_file_snapshots snapshot "
                        "JOIN library_versions version ON version.id = snapshot.version_id"
                    )
                }
            # Runs first: a matching durable catalog can reuse its existing snapshot.
            sources = [
                (path, "run")
                for path in sorted(Path(self.settings.runs_dir).glob("*/library.json"))
            ] + [
                (path, "catalog")
                for path in sorted(Path(self.settings.library_dir).glob("*/*/library.json"))
            ]
            for path, kind in sources:
                if self._stopped.is_set():
                    break
                key = str(path.resolve())
                cached = self._cache.get(key)
                previous = indexed.get(key)
                if previous is not None:
                    self._refresh_source_activity(path, previous["source_run_id"])
                if cached is None and previous is not None:
                    signature = tuple(
                        tuple(item) for item in json.loads(previous["signature_json"])
                    )
                    if signature:
                        cached = (
                            tuple(Path(item[0]) for item in signature),
                            signature,
                            previous["fingerprint"],
                        )
                        self._cache[key] = cached
                if (
                    cached is not None
                    and previous is not None
                    and cached[2] == previous["fingerprint"]
                    # Revisit snapshots created by an older report renderer once.
                    and all(dependency in cached[0]
                            for dependency in (Path(__file__), *renderer_files()))
                    and _signature(cached[0]) == cached[1]
                ):
                    continue
                # Warn about an unindexable save once; retry it after the file changes.
                saved = _signature((path,))
                if self._invalid.get(key) == saved:
                    continue
                self._invalid.pop(key, None)
                try:
                    self._refresh_file(path, kind, previous)
                except _InvalidLibrary as exc:
                    self._invalid[key] = saved
                    logger.warning("Unable to index local library %s: %s", path, exc)
                except Exception:
                    # A corrupt/in-progress save must not hide the other libraries or
                    # prevent the application from launching. Retry on the next pass.
                    logger.warning("Unable to index local library %s", path, exc_info=True)

    def _refresh_source_activity(self, path: Path, source_run_id: str) -> None:
        """Refresh ordering metadata independently of immutable report snapshots."""
        run_dir = path.parent
        if source_run_id and Path(source_run_id).name == source_run_id:
            candidate = Path(self.settings.runs_dir) / source_run_id
            if candidate.is_dir():
                run_dir = candidate
        source = run_dir / "library.json"
        if not source.is_file():
            source = path
        publication_path = run_dir / "publication.json"
        signature = _signature((source, publication_path))
        if self._activity_cache.get(source_run_id) == signature:
            return
        publication = _read_object(publication_path)
        published_at = (
            _publication_time(publication.get("updated_at"))
            if publication.get("state") == "published" else None
        )
        try:
            saved_at = datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(
                timespec="microseconds"
            )
        except OSError:
            return
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO local_source_activity (source_run_id, published_at, saved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_run_id) DO UPDATE SET
                    published_at = COALESCE(excluded.published_at, published_at),
                    saved_at = excluded.saved_at
                """,
                (source_run_id, published_at, saved_at),
            )
        self._activity_cache[source_run_id] = signature

    def pending_reports(self) -> int:
        with self.database.read() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM local_file_snapshots WHERE report_pending = 1"
            ).fetchone()[0]

    def _enrich_pending(self) -> None:
        with self.database.read() as connection:
            pending = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT snapshot.* FROM local_file_snapshots snapshot
                    JOIN library_versions version ON version.id = snapshot.version_id
                    LEFT JOIN local_source_activity activity
                        ON activity.source_run_id = version.source_run_id
                    WHERE snapshot.report_pending = 1
                    ORDER BY julianday(COALESCE(activity.published_at, activity.saved_at,
                                               version.created_at)) DESC, version.sequence DESC
                    """
                )
            ]
        for previous in pending:
            if self._stopped.is_set():
                break
            path = Path(previous["source_path"])
            try:
                with self._scan_lock:
                    self._refresh_file(path, previous["source_kind"], previous, prepare_report=True)
            except Exception:
                logger.warning("Unable to prepare saved report for %s", path, exc_info=True)
                with self.database.transaction() as connection:
                    connection.execute(
                        "UPDATE local_file_snapshots SET report_pending = 0, "
                        "signature_json = '[]' WHERE source_path = ?",
                        (previous["source_path"],),
                    )
                self._cache.pop(previous["source_path"], None)

    def _refresh_file(
        self, path: Path, kind: str, previous: dict | None, *, prepare_report: bool = False
    ) -> None:
        from .lazy_report import renderer_files

        watched: dict[Path, tuple] = {}
        for dependency in (
            Path(__file__),
            *renderer_files(),
            path,
            path.with_name("config.json"),
            path.with_name("library.provenance.json"),
            *(path.with_name(name) for name in _REPORT_NAMES),
        ):
            _watch(watched, dependency)
        try:
            library_bytes = _bounded_read(path, self.settings.max_local_library_bytes)
            payload = json.loads(library_bytes)
            model, benchmarks, function_count = validate_library_payload(payload)
        except ValueError as exc:
            raise _InvalidLibrary(str(exc)) from exc
        provenance = _read_object(path.with_name("library.provenance.json"))
        source_run_id = (
            path.parent.name if kind == "run" else str(provenance.get("source_run_id") or "")
        )
        run_config = path.with_name("config.json")
        if kind == "catalog" and source_run_id and Path(source_run_id).name == source_run_id:
            run_config = Path(self.settings.runs_dir) / source_run_id / "config.json"
            _watch(watched, run_config)
        run = _read_object(run_config)
        exact_benchmarks = run.get("benchmarks")
        if (
            isinstance(exact_benchmarks, list)
            and exact_benchmarks
            and all(isinstance(value, str) and value.strip() for value in exact_benchmarks)
        ):
            benchmarks = exact_benchmarks
        elif benchmarks == ["all"]:
            benchmarks = []
        identity = hashlib.sha256(
            json.dumps([str(path.resolve()), model, sorted(set(benchmarks))]).encode()
        ).hexdigest()[:32]
        branch_key = f"local-{kind}-{identity}"
        source_run_id = source_run_id or branch_key
        self._refresh_source_activity(path, source_run_id)
        references = normalize_reference_data(
            run.get("reference_data") or provenance.get("reference_data")
        )
        dataset_paths = run.get("parquet_paths") or []
        for dataset_path in dataset_paths:
            _watch(watched, Path(dataset_path).expanduser().resolve())
        if dataset_paths or "data_identity" not in provenance:
            data_identity = publication_data_identity(dataset_paths, reference_data=references)
        else:
            # A portable catalog can retain its saved source identity when the
            # original run configuration is no longer on this machine.
            try:
                data_identity = normalize_data_identity(provenance["data_identity"])
            except ValueError:
                data_identity = None
        # A durable catalog path is overwritten by later analyses. Bind its
        # snapshots to the analysis and source, rather than treating the path as
        # one permanent branch that cannot accept a different dataset.
        identity = hashlib.sha256(json.dumps(
            [str(path.resolve()), model, sorted(set(benchmarks)), source_run_id, data_identity],
            sort_keys=True,
        ).encode()).hexdigest()[:32]
        branch_key = f"local-{kind}-{identity}"
        if previous is not None:
            with self.database.read() as connection:
                prior = connection.execute(
                    "SELECT branch.branch_key, version.source_run_id, version.data_identity_json, "
                    "version.model_id, version.benchmarks_json "
                    "FROM library_versions version "
                    "JOIN library_branches branch ON branch.id = version.branch_id "
                    "WHERE version.id = ?", (previous["version_id"],),
                ).fetchone()
            # Run and durable copies can point at the same immutable version.
            # Either copy must advance that branch when its peer is temporarily
            # unreadable; the path kind does not change the analysis lineage.
            if (prior is not None
                    and prior["branch_key"].startswith(("local-run-", "local-catalog-"))
                    and prior["source_run_id"] == source_run_id
                    and prior["model_id"] == model
                    and sorted(json.loads(prior["benchmarks_json"])) == sorted(set(benchmarks))):
                prior_identity = normalize_data_identity(
                    json.loads(prior["data_identity_json"] or "null"),
                )
                if prior_identity is None or prior_identity == data_identity:
                    # Keep existing history, including the one-time binding of
                    # an older unknown-source branch to its newly known source.
                    branch_key = prior["branch_key"]
        metadata = UploadMetadata(
            source_run_id=source_run_id,
            model_id=model,
            benchmarks=benchmarks,
            harness=str(run.get("harness") or "unknown"),
            source_harness=str(
                run.get("source_harness") or provenance.get("source_harness") or "unknown"
            ),
            publish_trigger="local_catalog",
            data_identity=data_identity,
        )
        report, assets, saved_report, report_pending = self._report(
            path, payload, metadata, watched, defer_large=not prepare_report,
            prefer_source_links=(bool(references)
                                 and not any(local_viewer_base(r) for r in references)),
            reference_data=references,
        )
        if self._stopped.is_set():
            return
        fingerprint = hashlib.sha256(
            json.dumps(metadata.model_dump(), sort_keys=True).encode()
            + b"\0"
            + library_bytes
            + b"\0"
            + report
            + b"".join(name.encode() + hashlib.sha256(data).digest() for name, data in assets)
        ).hexdigest()
        key = str(path.resolve())
        watched_paths = tuple(watched)
        signature = _signature(watched_paths)
        # A save can replace files during indexing. Leave the cache invalid so the
        # next read revisits it, rather than accidentally caching the newer file stat.
        stable = signature == tuple(watched.values())
        with self.write_lock:
            if self._stopped.is_set():
                return
            if previous is None or previous["fingerprint"] != fingerprint:
                # The same save can exist both in a run and its durable catalog, or have
                # been published already. Reuse an equivalent latest version.
                version_id = self._matching_version(
                    metadata, payload, report, assets, saved_report
                )
                if version_id is None:
                    result = self.publish(
                        metadata=metadata,
                        branch_key=branch_key,
                        source_run_id=source_run_id,
                        known_library_id=None,
                        known_branch_id=None,
                        library_bytes=library_bytes,
                        report_bytes=report,
                        function_count=function_count,
                        report_assets=assets,
                    )
                    version_id = result["id"]
                with self.database.transaction() as connection:
                    connection.execute(
                        """
                        INSERT INTO local_file_snapshots
                        (source_path, fingerprint, signature_json, version_id,
                         report_pending, source_kind)
                    VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_path) DO UPDATE SET
                            fingerprint = excluded.fingerprint,
                        signature_json = excluded.signature_json,
                        version_id = excluded.version_id,
                        report_pending = excluded.report_pending,
                        source_kind = excluded.source_kind
                        """,
                        (
                            key,
                            fingerprint,
                            json.dumps(signature if stable else []),
                            version_id,
                            int(report_pending),
                            kind,
                        ),
                    )
            elif stable:
                with self.database.transaction() as connection:
                    connection.execute(
                        "UPDATE local_file_snapshots SET signature_json = ?, report_pending = ? "
                        "WHERE source_path = ?",
                        (json.dumps(signature), int(report_pending), key),
                    )
        if stable:
            self._cache[key] = (watched_paths, signature, fingerprint)
        else:
            self._cache.pop(key, None)

    def _matching_version(
        self,
        metadata: UploadMetadata,
        payload: dict,
        report: bytes,
        assets: list[tuple[str, bytes]],
        saved_report: bool,
    ) -> str | None:
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT version.id, version.library_key, version.report_sha256,
                    version.harness, version.source_harness, version.data_identity_json
                FROM library_versions version
                JOIN logical_libraries library ON library.id = version.library_id
                WHERE library.owner_id = 1 AND version.model_id = ?
                    AND version.benchmarks_json = ? AND version.source_run_id = ?
                ORDER BY version.sequence DESC
                LIMIT 1
                """,
                (
                    metadata.model_id,
                    json.dumps(metadata.benchmarks, ensure_ascii=False, separators=(",", ":")),
                    metadata.source_run_id,
                ),
            ).fetchall()
        for row in rows:
            try:
                if normalize_data_identity(json.loads(row["data_identity_json"] or "null")) != (
                    metadata.data_identity
                ):
                    continue
                if not same_library_content(
                    json.loads(self.store().get(row["library_key"])), payload,
                ):
                    continue
                # Reuse matching evidence even if this source already points at it.
                # A catalog copy or deferred large report must never replace its
                # enriched snapshot with a generated, trajectory-free placeholder.
                if not saved_report:
                    return row["id"]
                if (
                    row["report_sha256"] != hashlib.sha256(report).hexdigest()
                    or row["harness"] != metadata.harness
                    or row["source_harness"] != metadata.source_harness
                ):
                    continue
                with self.database.read() as connection:
                    stored_assets = {
                        asset["path"]: asset["sha256"]
                        for asset in connection.execute(
                            "SELECT path, sha256 FROM report_assets WHERE version_id = ?",
                            (row["id"],),
                        )
                    }
                if stored_assets == {
                    name: hashlib.sha256(content).hexdigest() for name, content in assets
                }:
                    return row["id"]
            except (OSError, ValueError):
                continue
        return None

    def _report(
        self,
        path: Path,
        payload: dict,
        metadata: UploadMetadata,
        watched: dict[Path, tuple],
        *,
        defer_large: bool = True,
        prefer_source_links: bool = False,
        reference_data: list[dict[str, str]] | None = None,
    ) -> tuple[bytes, list[tuple[str, bytes]], bool, bool]:
        from autofa.reports.recorded_relationships import read_state
        from autofa.session.status_log import latest_turn_status

        from .lazy_report import build_lazy_report

        relationship_state = read_state([path])
        run = path.parent
        source_run_id = metadata.source_run_id
        if (source_run_id and source_run_id not in {".", ".."}
                and Path(source_run_id).name == source_run_id and "\\" not in source_run_id):
            run = Path(self.settings.runs_dir) / source_run_id
        # Execution can change after the final library save. Watch even an absent
        # log so its first status invalidates the immutable dashboard snapshot.
        _watch(watched, run / "status_log.jsonl")
        agent_execution = latest_turn_status(run)

        lazy = build_lazy_report(
            FunctionLibrary.from_dict(payload), reference_data or [],
            source_harness=metadata.source_harness,
            relationship_state=relationship_state,
            agent_execution=agent_execution,
        )
        if lazy is not None:
            report, assets = lazy
            return report, assets, True, False
        report_pending = False
        candidates = []
        for name in _REPORT_NAMES:
            candidate = path.with_name(name)
            if candidate.is_file() and candidate.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                candidates.append(candidate)
        # Cloud inputs may use compact source links. Local stores retain embedded
        # evidence as well as verified viewer links. No network I/O here.
        candidates.sort(key=lambda item: (
            (item.name == "failure_cards_cloud.html") != prefer_source_links,
            -item.stat().st_mtime_ns,
        ))
        for candidate in candidates:
            try:
                if defer_large and candidate.stat().st_size > 1024 * 1024:
                    raise _DeferReport
                report = _bounded_read(candidate, self.settings.max_report_bytes)
                report.decode("utf-8")
                if b"<" not in report or b"\x00" in report:
                    raise ValueError("invalid local report")
                assets = self._report_assets(candidate, report, watched, defer_large=defer_large)
                report = _with_execution_status(report, agent_execution)
                return report, assets, True, False
            except _DeferReport:
                report_pending = True
                break
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # Background history maintenance must not interrupt the CLI wizard.
                # A failed fallback is still reported by the indexing worker.
                logger.debug(
                    "Unable to reuse saved report %s (%s); rendering saved library",
                    candidate, exc,
                )
        if agent_execution.get("status") != "unknown":
            retained = self._retained_report(metadata, payload)
            if retained is not None:
                report, assets = retained
                return _with_execution_status(report, agent_execution), assets, True, report_pending
        # Rendering uses saved scores and examples only. It executes no detectors and writes
        # no reports alongside the user's original files. A local import's examples are read
        # from its immutable store, in the deferred pass.
        from autofa.reports.failure_cards import build_data, render_html, spill_items

        dataset = _local_store_dataset(reference_data or [])
        if dataset is not None and defer_large:
            dataset, report_pending = None, True
        data = build_data(
            FunctionLibrary.from_dict(payload),
            dataset=dataset,
            source_harness=metadata.source_harness,
            reference_data=reference_data or [],
            prefer_source_links=False,
            relationship_state=relationship_state,
        )
        data["meta"]["agent_execution"] = agent_execution
        assets: list[tuple[str, bytes]] = []
        if dataset is not None:
            data, files = spill_items(data, _STORE_ITEMS_BASE)
            assets = [(f"{_STORE_ITEMS_BASE}/{name}", content) for name, content in files.items()]
        # Trajectories read from the store make this evidence, not a placeholder to fold away.
        return (render_html(data).encode("utf-8"), assets,
                bool(assets) or agent_execution.get("status") != "unknown", report_pending)

    def _retained_report(self, metadata: UploadMetadata, payload: dict):
        """Preserve enriched immutable evidence while refreshing execution status."""
        from .lazy_report import KEY_PREFIX, LazyReportAssets

        with self.database.read() as connection:
            row = connection.execute(
                "SELECT version.id, version.library_key, version.report_key, "
                "version.data_identity_json "
                "FROM library_versions version "
                "JOIN logical_libraries library ON library.id = version.library_id "
                "WHERE library.owner_id = 1 AND version.model_id = ? "
                "AND version.benchmarks_json = ? AND version.source_run_id = ? "
                "ORDER BY version.sequence DESC LIMIT 1",
                (metadata.model_id,
                 json.dumps(metadata.benchmarks, ensure_ascii=False, separators=(",", ":")),
                 metadata.source_run_id),
            ).fetchone()
            if row is None:
                return None
            assets = list(connection.execute(
                "SELECT path, artifact_key FROM report_assets WHERE version_id = ?",
                (row["id"],),
            ))
        try:
            if normalize_data_identity(json.loads(row["data_identity_json"] or "null")) != (
                metadata.data_identity
            ):
                return None
            store = self.store()
            if not same_library_content(json.loads(store.get(row["library_key"])), payload):
                return None
            # Reading a lazy descriptor must not materialize original trajectories.
            lazy = [asset["artifact_key"].startswith(KEY_PREFIX) for asset in assets]
            if any(lazy) and not all(lazy):
                return None
            files = [(asset["path"], store.get(asset["artifact_key"].removeprefix(KEY_PREFIX)))
                     for asset in assets]
            report = store.get(row["report_key"])
            return report, LazyReportAssets(files) if any(lazy) else files
        except (OSError, ValueError):
            return None

    def _report_assets(
        self,
        report_path: Path,
        report: bytes,
        watched: dict[Path, tuple],
        *,
        defer_large: bool = True,
    ) -> list[tuple[str, bytes]]:
        match = _REPORT_DATA.search(report)
        if match is None:
            return []
        data = json.loads(match.group(1))
        index = data.get("item_index") or {}
        base = (data.get("meta") or {}).get("items_base")
        if not index or not base:
            return []
        if not isinstance(index, dict) or len(index) > 10_000:
            raise ValueError("invalid local report asset index")
        if defer_large and len(index) > 50:
            raise _DeferReport
        root = report_path.parent.resolve()
        result = []
        total = 0
        for filename in sorted(set(index.values())):
            if self._stopped.is_set():
                break
            relative = PurePosixPath(f"{str(base).rstrip('/')}/{filename}")
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in str(relative)
                or relative.suffix != ".json"
            ):
                raise ValueError("invalid local report asset path")
            source = root.joinpath(*relative.parts).resolve()
            source.relative_to(root)
            _watch(watched, source)
            content = _bounded_read(source, self.settings.max_report_assets_unpacked_bytes)
            total += len(content)
            if total > self.settings.max_report_assets_unpacked_bytes:
                raise ValueError("local report assets exceed size limit")
            result.append((relative.as_posix(), content))
        return result


_STORE_ITEMS_BASE = "failure_cards_data/items"


def _local_store_dataset(reference_data: list[dict[str, str]]):
    """The dataset of the first intact local store a library was analysed from, or None."""
    from autofa.data.local_trace_store import resolve_store
    from autofa.data.parquet_store import Dataset

    for reference in reference_data:
        if reference.get("kind") != "local_evalrdl" or not reference.get("store_id"):
            continue
        try:
            path = resolve_store(str(reference["store_id"])) / "dataset.parquet"
        except (OSError, ValueError):
            continue
        if path.is_file():
            return Dataset(parquet_paths=[path])
    return None
