# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote evaluation API client + local AutoFA parquet cache."""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from autofa.data.gym_adapter import converted_root
from autofa.data.source_errors import SourceDataError as DataApiError
from autofa.defaults import default_value

DEFAULT_DATA_API_BASE_URL = default_value("data_api_base_url")
# The first versioned cache follows EvalRDL's uncapped, lane-aware export.
REMOTE_EXPORT_CACHE_VERSION = 2
log = logging.getLogger(__name__)
_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSPORT_ERRORS = (
    urllib.error.URLError, TimeoutError, ConnectionError,
    http.client.IncompleteRead, http.client.RemoteDisconnected,
)

# Catalog fields worth keeping in the sidecar for display / debugging.
_SIDECAR_RUN_KEYS = (
    "run_id",
    "run_uuid",
    "source_run_id",
    "label",
    "benchmark",
    "harness",
    "model",
    "run_ts",
    "n_rollouts",
    "cluster",
    "source_format",
    "binding_version",
)


class _ArtifactRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow object-store redirects while rebuilding requests without credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed, origin = urllib.parse.urlsplit(newurl), urllib.parse.urlsplit(req.full_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.fragment
                or (parsed.scheme == "http" and
                    (origin.scheme != "http" or parsed.netloc != origin.netloc))):
            raise DataApiError("Unsupported or insecure artifact download redirect", status=501)
        # Do not reuse Authorization, cookies, proxy credentials, or request headers
        # on signed redirects to storage services, including same-origin redirects.
        return urllib.request.Request(newurl, headers={"Accept": "application/octet-stream"})


class DataApiClient:
    """Thin GET client for the benchmark-run data layer."""

    def __init__(
        self, base_url: str | None = None, *,
        timeout: float = 120.0, max_attempts: int = 6,
    ):
        if base_url is None:
            base_url = default_value("data_api_base_url")
        from autofa.services import validate_service_address

        self.base_url = validate_service_address(base_url, name="EvalRDL remote server")
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)

    def _request_with_retries(
        self, operation: Callable[[], Any], *, label: str,
        on_retry: Callable[[int, float], None] | None = None,
        redact_errors: bool = False,
    ) -> Any:
        """Retry complete GETs; optionally report backoff and redact signed-URL errors."""
        for attempt in range(self.max_attempts):
            retry_after = None
            try:
                return operation()
            except urllib.error.HTTPError as exc:
                status = exc.code
                retryable = status in _RETRYABLE_HTTP_STATUS
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    detail = "" if redact_errors else exc.read(300).decode(
                        "utf-8", errors="replace",
                    )
                except _TRANSPORT_ERRORS:
                    detail = ""
                finally:
                    exc.close()
                reason = f"HTTP {status}" if redact_errors else exc.reason
                error = DataApiError(f"API {status} {label}: {detail or reason}",
                                     status=status)
                cause = None if redact_errors else exc
            except _TRANSPORT_ERRORS as exc:
                reason = getattr(exc, "reason", exc)
                retryable = not isinstance(reason, ssl.SSLCertVerificationError)
                detail = reason
                if redact_errors:
                    detail = type(reason if isinstance(reason, BaseException) else exc).__name__
                error = DataApiError(f"Network error {label}: {detail}")
                cause = None if redact_errors else exc
            if not retryable or attempt + 1 == self.max_attempts:
                raise error from cause
            delay = _retry_delay(attempt, retry_after)
            log.warning("%s; retrying (attempt %d/%d) in %.1fs",
                        error, attempt + 2, self.max_attempts, delay)
            if on_retry:
                on_retry(attempt + 2, delay)
            time.sleep(delay)

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        if not self.base_url:
            from autofa.services import EVAL_NOT_CONFIGURED

            raise DataApiError(EVAL_NOT_CONFIGURED)
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"
        if query:
            # The unnamed conversation lane is selected with an explicit lane=.
            # Omitting it requests every lane, including verdict/model-call metadata.
            pairs = [(k, v) for k, v in query.items()
                     if v is not None and (v != "" or k == "lane")]
            if pairs:
                url = f"{url}?{urllib.parse.urlencode(pairs, doseq=True)}"
        return url

    def _get_json(
        self, path: str, query: dict[str, Any] | None = None, *,
        on_retry: Callable[[int, float], None] | None = None,
    ) -> Any:
        url = self._url(path, query)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        def read_body():
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()

        body = self._request_with_retries(
            read_body, label=f"fetching {path}", on_retry=on_retry,
        )
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DataApiError(f"Invalid JSON from {url}: {e}") from e

    def query_analytics(self, sql: str) -> dict[str, Any]:
        """Run one read-only SQL statement against the analytics tables."""
        url = self._url("/v1/analytics/query")
        req = urllib.request.Request(
            url, data=json.dumps({"sql": sql}).encode("utf-8"), method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

        def read_body():
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()

        body = self._request_with_retries(read_body, label="querying analytics")
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DataApiError(f"Invalid JSON from {url}: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise DataApiError("Expected rows from /v1/analytics/query")
        return data

    def meta(self) -> dict[str, Any]:
        data = self._get_json("/v1/meta")
        if not isinstance(data, dict):
            raise DataApiError("Expected object from /v1/meta")
        return data

    def list_runs(self, **filters: Any) -> list[dict[str, Any]]:
        data = self._get_json("/v1/runs", filters or None)
        if not isinstance(data, list):
            raise DataApiError("Expected array from /v1/runs")
        return [r for r in data if isinstance(r, dict)]

    def get_run(self, run_id: str) -> dict[str, Any]:
        data = self._get_json(f"/v1/runs/{urllib.parse.quote(run_id, safe='')}")
        if not isinstance(data, dict):
            raise DataApiError(f"Expected object from /v1/runs/{run_id}")
        return data

    @staticmethod
    def _rollout_identity(run_id: str, task_id: str, repeat: str | int) -> str:
        return "/".join(
            urllib.parse.quote(str(value), safe="") for value in (run_id, task_id, repeat)
        )

    def get_trace_resource(
        self, run_id: str, task_id: str, repeat: int, resource: str,
    ) -> Any:
        """Read an optional source resource not supplied by every bulk-export version."""
        routes = {"provenance": "provenance", "relations": "edges", "prompt_summaries": "prompts"}
        if resource not in routes:
            raise ValueError(f"unsupported execution resource {resource!r}")
        identity = self._rollout_identity(run_id, task_id, repeat)
        return self._get_json(f"/v1/rollouts/{identity}/{routes[resource]}")

    def _rollout_json(
        self, run_id: str, task_id: str, repeat: str | int, suffix: str = "", *,
        expect: type = dict,
    ) -> Any:
        identity = self._rollout_identity(run_id, task_id, repeat)
        path = f"/v1/rollouts/{identity}{suffix}"
        data = self._get_json(path)
        if not isinstance(data, expect):
            raise DataApiError(f"Expected {'object' if expect is dict else 'array'} from {path}")
        return data

    def get_rollout_artifacts(
        self, run_id: str, task_id: str, repeat: str | int,
    ) -> list[dict[str, Any]]:
        """Return artifact metadata associated with one rollout."""
        data = self._rollout_json(run_id, task_id, repeat, "/artifacts", expect=list)
        return [artifact for artifact in data if isinstance(artifact, dict)]

    def get_rollout_artifact_texts(
        self,
        artifacts: list[dict[str, Any]],
        *,
        run_id: str,
        task_id: str,
        repeat: str | int,
    ) -> list[dict[str, Any]]:
        """Return caller-selected artifacts with text, or ``text_error`` when unreadable."""
        hydrated = []
        identity = self._rollout_identity(run_id, task_id, repeat)
        for artifact in artifacts:
            item = dict(artifact)
            item.pop("text", None)
            text_url = item.get("text_url")
            try:
                artifact_idx = item.get("artifact_idx")
                if type(artifact_idx) is not int:
                    raise DataApiError("Artifact does not expose an integer artifact_idx")
                expected_url = f"/v1/rollouts/{identity}/artifacts/{artifact_idx}/text"
                if text_url != expected_url:
                    raise DataApiError("Artifact text_url does not match its rollout identity")
                expected_sha256 = item.get("sha256")
                expected_size = item.get("size_bytes")
                if (
                    not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                    or type(expected_size) is not int
                    or expected_size < 0
                ):
                    raise DataApiError("Artifact text metadata lacks verified size and SHA-256")
                data = self._get_json(text_url)
                if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                    raise DataApiError(f"Expected text object from {text_url}")
                encoded = data["text"].encode("utf-8")
                if len(encoded) != expected_size:
                    raise DataApiError("Artifact text size does not match its metadata")
                if hashlib.sha256(encoded).hexdigest() != expected_sha256:
                    raise DataApiError("Artifact text SHA-256 does not match its metadata")
                item["text"] = data["text"]
            except DataApiError as exc:
                item["text_error"] = str(exc)
            hydrated.append(item)
        return hydrated

    def get_rollout_verifier_artifacts(
        self,
        artifacts: list[dict[str, Any]],
        *,
        run_id: str,
        task_id: str,
        repeat: str | int,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper returning hydrated verifier artifacts only."""
        return self.get_rollout_artifact_texts(
            [artifact for artifact in artifacts if artifact.get("role") == "verifier"],
            run_id=run_id,
            task_id=task_id,
            repeat=repeat,
        )

    def read_raw_record(
        self, run_id: str, source_ref: str, *, max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Read one potentially large JSONL record at ``path:offset``."""
        path, _, raw_offset = source_ref.rpartition(":")
        try:
            offset = int(raw_offset)
        except ValueError as exc:
            raise DataApiError(f"invalid raw record ref {source_ref!r}") from exc
        endpoint = f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/raw/slice"
        chunks: list[str] = []
        consumed = 0
        while consumed < max_bytes:
            length = min(1024 * 1024, max_bytes - consumed)
            data = self._get_json(
                endpoint, {"path": path, "offset": offset + consumed, "length": length},
            )
            if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                raise DataApiError(f"Expected text from {endpoint}")
            chunks.append(data["text"])
            if "\n" in data["text"] or data.get("eof"):
                try:
                    record = json.loads("".join(chunks).split("\n", maxsplit=1)[0])
                except json.JSONDecodeError as exc:
                    raise DataApiError(f"invalid raw record at {source_ref}: {exc}") from exc
                if not isinstance(record, dict):
                    raise DataApiError(f"raw record at {source_ref} is not an object")
                return record
            chunk_bytes = int(data.get("length") or len(data["text"].encode("utf-8")))
            if chunk_bytes <= 0:
                break
            consumed += chunk_bytes
        raise DataApiError(f"raw record at {source_ref} exceeds {max_bytes} bytes")

    def download_artifact(
        self, run_id: str, task_id: str, repeat: int, artifact_idx: int, dest: Path,
    ) -> Path:
        """Stream one catalog-identified artifact, safely following storage redirects.

        The caller verifies catalog size/digest before promoting these bytes to
        shared evidence. No artifact URI or filename is accepted by this method.
        """
        base = urllib.parse.urlsplit(self.base_url)
        if (base.scheme not in {"http", "https"} or not base.hostname or base.username
                or base.password or base.query or base.fragment):
            raise DataApiError("Invalid configured artifact API origin", status=501)
        if (type(repeat) is not int or repeat < 0 or type(artifact_idx) is not int
                or artifact_idx < 0 or any(not isinstance(value, str) or not value
                                          or value in {".", ".."}
                                          for value in (run_id, task_id))):
            raise ValueError("Invalid artifact source identity")
        identity = self._rollout_identity(run_id, task_id, repeat)
        path = f"/v1/rollouts/{identity}/artifacts/{artifact_idx}/download"
        request = urllib.request.Request(self._url(path),
                                         headers={"Accept": "application/octet-stream"})
        opener = urllib.request.build_opener(_ArtifactRedirectHandler())
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".partial")

        def download():
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    raw_size = response.headers.get("Content-Length")
                    expected = int(raw_size) if raw_size and raw_size.isdigit() else None
                    size = 0
                    with partial.open("wb") as handle:
                        while chunk := response.read(256 * 1024):
                            handle.write(chunk)
                            size += len(chunk)
                    if expected is not None and size < expected:
                        raise http.client.IncompleteRead(b"", expected - size)
                    if expected is not None and size > expected:
                        raise DataApiError("Artifact exceeded response Content-Length")
            except BaseException:
                partial.unlink(missing_ok=True)
                raise

        try:
            self._request_with_retries(download, label="downloading recorded artifact")
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
        return dest

    def download_autofa_export(
        self,
        run_id: str,
        dest: Path,
        *,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        """Stream ``GET /v1/runs/{run_id}/export/autofa`` to ``dest``.

        ``progress(downloaded_bytes, total_bytes_or_none)`` is called as chunks
        arrive. ``total`` comes from ``Content-Length`` when the server sends it.
        """
        return self._download_export(run_id, "autofa", dest, progress=progress)

    def download_trace_export(
        self,
        run_id: str,
        dest: Path,
        *,
        progress: Callable[[int, int | None], None] | None = None,
        rollouts: list[tuple[str, int]] | None = None,
    ) -> Path:
        """Download the run's trace tables ZIP, restarting interrupted transfers."""
        if rollouts is not None and not 1 <= len(rollouts) <= 500:
            raise ValueError("Trace export selects between 1 and 500 rollouts")
        query = {"rollout": [f"{task}:{repeat}" for task, repeat in rollouts]} if rollouts else None
        return self._download_export(run_id, "trace", dest, progress=progress, query=query)

    def _download_export(
        self, run_id: str, name: str, dest: Path, *,
        progress: Callable[[int, int | None], None] | None,
        query: dict[str, Any] | None = None,
    ) -> Path:
        url = self._url(f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/export/{name}", query)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        media_type = "application/vnd.apache.parquet" if name == "autofa" else "application/zip"
        req = urllib.request.Request(url, headers={"Accept": media_type + ",*/*"})

        def download():
            # Every retry restarts the response and byte counter; never append to
            # a partial export or make it visible as a reusable cache entry.
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                total: int | None = None
                headers = getattr(resp, "headers", None)
                if headers is not None:
                    raw_len = headers.get("Content-Length") or headers.get("content-length")
                    if raw_len and str(raw_len).isdigit():
                        total = int(raw_len)
                downloaded = 0
                if progress is not None:
                    progress(downloaded, total)
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(downloaded, total)
                if total is not None and downloaded < total:
                    raise http.client.IncompleteRead(b"", total - downloaded)
                if total is not None and downloaded > total:
                    raise DataApiError(f"Export for {run_id} exceeded Content-Length")
            if not downloaded:
                raise DataApiError(f"Empty {name} export for run {run_id}")
            # Streamed responses have no Content-Length. Check the container's
            # footer before replacing a complete cached download.
            try:
                if name == "autofa":
                    import pyarrow.parquet as pq

                    with pq.ParquetFile(tmp):
                        pass
                else:
                    with zipfile.ZipFile(tmp) as archive:
                        if "manifest.json" not in archive.namelist():
                            raise ValueError("Trace archive lacks manifest.json")
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise DataApiError(f"Invalid {name} export for run {run_id}: {exc}") from exc

        def attempt_download():
            try:
                download()
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

        try:
            self._request_with_retries(
                attempt_download, label=f"downloading {name} export for {run_id}",
            )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    delay = min(60.0, 5.0 * 2 ** min(attempt, 4))
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return delay
        if seconds >= 0:
            delay = max(delay, min(60.0, seconds))
    return delay


def remote_cache_dir(
    run_id: str,
    *,
    base_url: str | None = None,
) -> Path:
    """Return an origin-scoped cache directory for one Data API run."""
    if base_url is None:
        base_url = default_value("data_api_base_url")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    origin_key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:12]
    run_key = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    root = converted_root() / "remote" / origin_key / f"{safe[:80]}-{run_key}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def materialize_remote_run(
    run_id: str,
    *,
    base_url: str | None = None,
    force: bool = False,
    client: DataApiClient | None = None,
    catalog: dict[str, Any] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download (or reuse) AutoFA parquet for ``run_id`` under converted/remote/.

    Returns an immutable generation's ``dataset.parquet``. Refreshing the
    current generation never overwrites a path referenced by a saved analysis.
    ``progress`` is forwarded to the download when a fetch is needed.
    """
    if base_url is None:
        base_url = default_value("data_api_base_url")
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id is required")

    api = client or DataApiClient(base_url)
    canonical_id = (catalog or {}).get("run_id")
    if isinstance(canonical_id, str) and canonical_id.strip():
        run_id = canonical_id.strip()
    if not force:
        cached = current_remote_export(run_id, base_url=api.base_url)
        if cached is not None:
            return cached
    run_meta = catalog
    if run_meta is None:
        try:
            run_meta = api.get_run(run_id)
        except DataApiError as e:
            if e.status != 404:
                raise
            run_meta = {"run_id": run_id}
    canonical_id = run_meta.get("run_id")
    if isinstance(canonical_id, str) and canonical_id.strip():
        run_id = canonical_id.strip()
    if not force:
        cached = current_remote_export(run_id, base_url=api.base_url)
        if cached is not None:
            return cached
    cache = remote_cache_dir(run_id, base_url=api.base_url)
    generation = uuid.uuid4().hex
    generation_root = cache / "exports" / generation
    generation_root.mkdir(parents=True)
    out_parquet = generation_root / "dataset.parquet"
    pointer_temp = cache / f"current-{generation}.partial"
    publication_started = False
    try:
        api.download_autofa_export(run_id, out_parquet, progress=progress)
        _write_remote_source_sidecar(
            generation_root / "source.json", run_id=run_id, api=api,
            run_meta=run_meta, generation=generation,
        )
        # Publish only after both data and provenance are complete. Concurrent
        # refreshes have distinct generations; the last completed one is current.
        pointer_temp.write_text(json.dumps({"generation": generation}) + "\n", encoding="utf-8")
        publication_started = True
        pointer_temp.replace(cache / "current.json")
    except BaseException:
        # A signal can arrive after the atomic rename but before it returns.
        # Once the pointer moved, readers may already hold this generation.
        if not publication_started or pointer_temp.exists():
            shutil.rmtree(generation_root, ignore_errors=True)
        raise
    finally:
        pointer_temp.unlink(missing_ok=True)
    return out_parquet


class MissingSelectedRolloutsError(DataApiError):
    """A cached export does not contain every selected rollout."""


def stage_export_subset(
    source: Path,
    rollouts: list[tuple[str, int]],
    *,
    destination_dir: Path | None = None,
) -> Path:
    """Write selected export rows in selection order, reusing an export already in that order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    selected = [(str(task), int(repeat)) for task, repeat in rollouts]
    if not selected:
        raise ValueError("rollout selection must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("rollout selection contains duplicates")
    stat = source.stat()
    key = json.dumps([str(source.resolve()), stat.st_size, stat.st_mtime_ns, selected])
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    parent = Path(destination_dir) if destination_dir is not None else source.parent
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{source.stem}.selection-{digest}{source.suffix}"
    if destination.is_file():
        try:
            cached = pq.read_table(destination, columns=["problem_id", "run_idx"]).to_pylist()
            if [(str(row["problem_id"]), int(row["run_idx"])) for row in cached] == selected:
                return destination
        except (OSError, ValueError, TypeError, KeyError):
            pass

    parquet = pq.ParquetFile(source)
    if not {"problem_id", "run_idx"} <= set(parquet.schema_arrow.names):
        raise DataApiError("AutoFA export lacks rollout identity columns")
    wanted = set(selected)
    locations: dict[tuple[str, int], tuple[int, int]] = {}
    for group in range(parquet.metadata.num_row_groups):
        rows = parquet.read_row_group(group, columns=["problem_id", "run_idx"]).to_pylist()
        for offset, row in enumerate(rows):
            identity = (str(row["problem_id"]), int(row["run_idx"]))
            if identity not in wanted:
                continue
            if identity in locations:
                raise DataApiError(f"AutoFA export has duplicate rollout {identity!r}")
            locations[identity] = (group, offset)
    missing = [identity for identity in selected if identity not in locations]
    if missing:
        raise MissingSelectedRolloutsError(
            f"AutoFA export lacks {len(missing)} selected rollout(s), e.g. {missing[0]!r}; "
            "refresh the export and retry"
        )
    ordered = [locations[identity] for identity in selected]
    if len(selected) == parquet.metadata.num_rows and ordered == sorted(ordered):
        return source
    groups = {group: parquet.read_row_group(group) for group, _ in ordered}
    table = pa.concat_tables(groups[group].slice(offset, 1) for group, offset in ordered)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        pq.write_table(table, temp, compression="zstd")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination


def current_remote_export(run_id: str, *, base_url: str) -> Path | None:
    """Resolve the current complete export generation, if any."""
    cache = remote_cache_dir(run_id, base_url=base_url)
    try:
        pointer = json.loads((cache / "current.json").read_text(encoding="utf-8"))
        generation = pointer.get("generation") if isinstance(pointer, dict) else None
        if (isinstance(generation, str) and len(generation) == 32
                and all(c in "0123456789abcdef" for c in generation)):
            path = cache / "exports" / generation / "dataset.parquet"
            if remote_export_is_current(path, run_id=run_id, base_url=base_url):
                return path
    except (OSError, ValueError):
        pass
    return None


def _write_remote_source_sidecar(
    sidecar: Path,
    *,
    run_id: str,
    api: DataApiClient,
    run_meta: dict[str, Any],
    generation: str,
) -> None:
    payload = {
        "export_cache_version": REMOTE_EXPORT_CACHE_VERSION,
        "export_generation": generation,
        "run_id": run_id,
        "base_url": api.base_url.rstrip("/"),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "export_path": (
            f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/export/autofa"
        ),
        "catalog": {k: run_meta.get(k) for k in _SIDECAR_RUN_KEYS if k in run_meta},
    }
    stat = (sidecar.parent / "dataset.parquet").stat()
    payload["export_size"] = stat.st_size
    payload["export_mtime_ns"] = stat.st_mtime_ns
    temp = sidecar.with_suffix(".partial")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(sidecar)


def remote_export_is_current(path: Path, *, run_id: str, base_url: str) -> bool:
    """Only reuse this client's versioned export from the requested source.

    Older downloads may contain the upstream 50k reasoning cap. Never upgrade
    their sidecar without downloading the corrected bytes. The generation also
    invalidates reconstructed trace caches after an explicit refresh.
    """
    try:
        stat = path.stat()
        meta = json.loads((path.parent / "source.json").read_text(encoding="utf-8"))
        return (
            isinstance(meta, dict)
            and meta.get("export_cache_version") == REMOTE_EXPORT_CACHE_VERSION
            and bool(meta.get("export_generation"))
            and meta.get("run_id") == run_id
            and meta.get("base_url") == base_url.rstrip("/")
            and stat.st_size > 0
            and meta.get("export_size") == stat.st_size
            and meta.get("export_mtime_ns") == stat.st_mtime_ns
        )
    except (OSError, ValueError):
        return False
