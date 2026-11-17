# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import stat
import threading
import uuid
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import urlencode

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from .auth import AuthContext, SessionManager
from .catalog import LocalCatalog
from .config import Settings
from .db import Database
from .repository import (
    BranchIdentityConflictError,
    LibraryRepository,
    SelectorNotFoundError,
    to_record,
)
from .schemas import (
    BranchPage,
    BranchRecord,
    LibraryPage,
    LibraryRecord,
    MeResponse,
    PublicationMetadata,
    PublicationRecord,
    UploadMetadata,
    VersionPage,
    VersionRecord,
    validate_library_payload,
)
from .storage import ArtifactStore, create_artifact_store
from .theme import ui_theme

logger = logging.getLogger("autofa_library_server")

PACKAGE_DIR = Path(__file__).parent
REPORT_CSP = (
    "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; img-src data:; connect-src 'self'; "
    "font-src data:; media-src data:; object-src 'none'; frame-src 'none'; "
    "base-uri 'none'; form-action 'none'"
)
REPORT_DATA_PATTERN = re.compile(
    rb'(<script id="autofa-data" type="application/json">)(.*?)(</script>)',
    flags=re.DOTALL,
)
MAX_REPORT_ASSET_FILES = 10_000
REPORT_VIEW_VERSION = "7"

bearer = HTTPBearer(
    auto_error=False,
    scheme_name="LocalToken",
    description="Token from the local AutoFA profile",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _report_response(
    content: bytes,
    *,
    immutable: bool = False,
    public: bool = False,
    etag: str | None = None,
    status_code: int = 200,
) -> Response:
    cache_control = "no-store"
    if immutable and not public:
        cache_control = "private, max-age=31536000, immutable"
    headers = {
        "Cache-Control": cache_control,
        "Content-Security-Policy": REPORT_CSP,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }
    if etag:
        headers["ETag"] = f'"{etag}"'
    return Response(
        content,
        status_code=status_code,
        media_type="text/html; charset=utf-8",
        headers=headers,
    )


def _etag_matches(request: Request, etag: str) -> bool:
    expected = f'"{etag}"'
    return any(
        candidate.strip() in {"*", expected, f"W/{expected}"}
        for candidate in request.headers.get("if-none-match", "").split(",")
    )


def _report_asset_response(content: bytes) -> Response:
    return Response(
        content,
        media_type="application/json",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


def _report_with_assets_base(content: bytes, asset_url_prefix: str) -> bytes:
    match = REPORT_DATA_PATTERN.search(content)
    if match is None:
        return content
    try:
        data = json.loads(match.group(2))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content
    if not (data.get("item_index") and (data.get("meta") or {}).get("items_base")):
        return content
    data = dict(data)
    meta = dict(data.get("meta") or {})
    original_base = str(meta["items_base"]).strip("/")
    meta["items_base"] = f"{asset_url_prefix.rstrip('/')}/{original_base}"
    data["meta"] = meta
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    return content[: match.start(2)] + payload.encode("utf-8") + content[match.end(2) :]


def _report_with_current_template(content: bytes, library_content: bytes) -> bytes:
    """Display saved AutoFA reports with the current UI, retaining stored evidence."""
    match = REPORT_DATA_PATTERN.search(content)
    if match is None:
        return content
    from autofa.core.library import FunctionLibrary
    from autofa.reports.failure_cards import refresh_report_data, render_html

    try:
        data = json.loads(match.group(2))
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("meta"), dict)
            or not isinstance(data.get("functions"), list)
            or not all(isinstance(fn, dict) for fn in data["functions"])
        ):
            return content
        library = FunctionLibrary.from_dict(json.loads(library_content))
        return render_html(refresh_report_data(data, library)).encode("utf-8")
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.warning("Unable to refresh report presentation; keeping saved HTML")
        return content


def _report_with_saved_efficiency(content: bytes, saved_content: bytes) -> bytes:
    """Retain uploaded efficiency metrics when replacing a report with lazy traces."""
    match = REPORT_DATA_PATTERN.search(saved_content)
    if match is None:
        return content
    try:
        meta = json.loads(match.group(2)).get("meta") or {}
        efficiency = meta.get("efficiency_metrics")
    except (ValueError, AttributeError):
        return content
    if not isinstance(efficiency, dict):
        return content
    from autofa.reports.efficiency_card import _inject_overview

    return _inject_overview(
        content.decode("utf-8"), efficiency,
        sections=efficiency.get("sections") or {},
        layout=meta.get("efficiency_layout") or "replace",
    ).encode("utf-8")


def _require_content_type(
    upload: UploadFile,
    allowed: set[str],
    label: str,
) -> None:
    content_type = (
        (upload.content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    )
    if content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{label} has unsupported content type {content_type}",
        )


async def _bounded_upload(upload: UploadFile, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(min(1024 * 1024, maximum + 1)):
        size += len(chunk)
        if size > maximum:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"{label} exceeds the {maximum}-byte limit",
            )
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=422, detail=f"{label} cannot be empty")
    return b"".join(chunks)


def _parse_publication_uploads(
    settings: Settings,
    metadata_raw: str,
    library_bytes: bytes,
    report_bytes: bytes,
) -> tuple[PublicationMetadata, dict, int]:
    if len(metadata_raw.encode("utf-8")) > settings.max_metadata_bytes:
        raise HTTPException(status_code=413, detail="metadata exceeds the configured limit")
    try:
        raw = json.loads(metadata_raw)
        if isinstance(raw, str):
            raw = json.loads(raw)
        metadata = PublicationMetadata.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid metadata: {exc}") from exc

    try:
        library_payload = json.loads(library_bytes.decode("utf-8"))
        library_model, _, function_count = validate_library_payload(library_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid library JSON: {exc}") from exc
    if metadata.model_id != library_model:
        raise HTTPException(status_code=422, detail="metadata and library model_id differ")

    try:
        report_text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="failure-card must be UTF-8 HTML") from exc
    if "\x00" in report_text or "<" not in report_text or ">" not in report_text:
        raise HTTPException(status_code=422, detail="failure-card is not valid HTML content")
    return metadata, library_payload, function_count


def _safe_report_asset_path(value: str) -> str:
    if not value or "\\" in value:
        raise HTTPException(status_code=422, detail="report asset path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".json"
    ):
        raise HTTPException(status_code=422, detail="report asset path is invalid")
    return path.as_posix()


def _expected_report_asset_paths(report_bytes: bytes) -> set[str]:
    match = REPORT_DATA_PATTERN.search(report_bytes)
    if match is None:
        raise HTTPException(
            status_code=422,
            detail="failure-card assets require embedded autofa-data",
        )
    try:
        data = json.loads(match.group(2))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="failure-card has invalid embedded autofa-data",
        ) from exc
    index = data.get("item_index") or {}
    items_base = (data.get("meta") or {}).get("items_base")
    if not isinstance(index, dict) or not index or not items_base:
        raise HTTPException(
            status_code=422,
            detail="failure-card does not reference trajectory sidecars",
        )
    return {
        _safe_report_asset_path(f"{str(items_base).rstrip('/')}/{filename}")
        for filename in index.values()
    }


def _parse_report_assets(
    settings: Settings,
    report_bytes: bytes,
    archive_bytes: bytes,
) -> list[tuple[str, bytes]]:
    expected = _expected_report_asset_paths(report_bytes)
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=422,
            detail="failure-card assets are not a valid ZIP archive",
        ) from exc

    with archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if len(files) > MAX_REPORT_ASSET_FILES:
            raise HTTPException(
                status_code=413,
                detail="failure-card assets contain too many files",
            )
        total_size = sum(info.file_size for info in files)
        if total_size > settings.max_report_assets_unpacked_bytes:
            raise HTTPException(
                status_code=413,
                detail="unpacked failure-card assets exceed the configured limit",
            )

        result: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for info in files:
            mode = info.external_attr >> 16
            if info.flag_bits & 0x1 or stat.S_ISLNK(mode):
                raise HTTPException(
                    status_code=422,
                    detail="failure-card assets contain an unsupported ZIP entry",
                )
            path = _safe_report_asset_path(info.filename)
            if path in seen or path not in expected:
                raise HTTPException(
                    status_code=422,
                    detail="failure-card assets do not match the report index",
                )
            payload = archive.read(info)
            try:
                decoded = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"failure-card asset {path} is not valid JSON",
                ) from exc
            if not isinstance(decoded, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"failure-card asset {path} must contain a JSON object",
                )
            result.append((path, payload))
            seen.add(path)

    if seen != expected:
        raise HTTPException(
            status_code=422,
            detail="failure-card assets are missing report-indexed trajectories",
        )
    return result


def _to_version_record(item: dict) -> VersionRecord:
    base = (
        f"/api/v2/libraries/{item['library_id']}/branches/{item['branch_id']}/versions/{item['id']}"
    )
    return VersionRecord.model_validate(
        {
            **item,
            "library_url": f"{base}/library.json",
            "failure_card_url": f"{base}/failure-card",
        }
    )


def _to_publication_record(item: dict) -> PublicationRecord:
    version = _to_version_record(item)
    return PublicationRecord(
        library_id=version.library_id,
        branch_id=version.branch_id,
        version_id=version.id,
        branch_key=version.branch_key,
        version_number=version.version_number,
        source_run_id=version.source_run_id,
        model_id=version.model_id,
        benchmarks=version.benchmarks,
        function_count=version.function_count,
        created_at=version.created_at,
        library_sha256=version.library_sha256,
        report_sha256=version.report_sha256,
        library_url=version.library_url,
        failure_card_url=version.failure_card_url,
        harness=version.harness,
        source_harness=version.source_harness,
        reference_data=version.reference_data,
        data_identity=version.data_identity,
        publish_trigger=version.publish_trigger,
        is_public=version.is_public,
    )


def create_app(settings: Settings) -> FastAPI:  # noqa: C901
    from autofa.local_service import validate_service_url

    validate_service_url(settings.public_url)
    database = Database(settings.database_path)
    repository = LibraryRepository(database)
    sessions = SessionManager(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        if application.state.artifact_store is None:
            application.state.artifact_store = create_artifact_store(settings)
        application.state.local_catalog.start()
        try:
            yield
        finally:
            await run_in_threadpool(application.state.local_catalog.close)

    application = FastAPI(
        title="AutoFA Local Library Server",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)
    application.state.settings = settings
    application.state.database = database
    application.state.repository = repository
    application.state.sessions = sessions
    application.state.artifact_store = None
    application.state.write_lock = threading.RLock()
    sessions.mount_routes(application)
    report_asset_secret = (settings.session_secret or secrets.token_urlsafe(32)).encode("utf-8")
    from .lazy_report import renderer_fingerprint

    # A saved descriptor resolves through the current navigation implementation.
    # Refresh browser caches when that implementation changes, including servers
    # with a stable session secret and reports pushed from another machine.
    report_renderer = renderer_fingerprint()

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    templates.env.globals["ui_theme"] = ui_theme
    templates.env.filters["display_time"] = lambda value: (
        datetime.fromisoformat(str(value)).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_DIR / "static")),
        name="static",
    )
    from .trace import mount_traces

    mount_traces(application, identity=lambda request: sessions.from_request(request).identity)

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        from autofa.local_service import valid_local_request

        if not valid_local_request(request, settings.public_url, settings.remote_hosts):
            return JSONResponse({"detail": "Local origin required"}, status_code=403)
        blocked = sessions.protect_request(request)
        if blocked is not None:
            return blocked
        if (
            request.method == "GET"
            and request.headers.get("X-AutoFA-Catalog-Poll") != "1"
            and (
                request.url.path in {"/", "/api/v2/libraries"}
                or re.fullmatch(r"/libraries/[^/]+", request.url.path)
            )
        ):
            application.state.local_catalog.request_refresh()
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response

    def report_asset_token(version_id: str) -> str:
        digest = hmac.new(
            report_asset_secret,
            f"report-assets:{version_id}:{report_renderer}".encode(),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def require_auth(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)] = None,
    ) -> AuthContext:
        if credentials is not None:
            if not secrets.compare_digest(credentials.credentials, settings.local_token):
                raise HTTPException(status_code=401, detail="Invalid local library token")
            return AuthContext(identity=sessions.identity, method="local_token")
        context = sessions.from_request(request)
        if context is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return context

    require_auth_dependency = Depends(require_auth)

    def optional_auth(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)] = None,
    ) -> AuthContext | None:
        if credentials is None and sessions.from_request(request) is None:
            return None
        return require_auth(request, credentials)

    optional_auth_dependency = Depends(optional_auth)

    def require_mutation_auth(
        request: Request,
        context: AuthContext = require_auth_dependency,
    ) -> AuthContext:
        if context.method == "session":
            supplied = request.headers.get("X-CSRF-Token", "")
            if not supplied or not secrets.compare_digest(supplied, context.csrf_token):
                raise HTTPException(status_code=403, detail="CSRF validation failed")
        return context

    require_mutation_auth_dependency = Depends(require_mutation_auth)

    from .trace_transfer import mount_transfer_routes

    mount_transfer_routes(application, settings, require_mutation_auth_dependency)

    def browser_context(request: Request) -> AuthContext:
        return sessions.from_request(request)

    def owned_or_404(context: AuthContext, library_id: str) -> dict:
        item = repository.get_owned(context.identity.id, library_id)
        if item is None:
            raise HTTPException(status_code=404, detail="library not found")
        return item

    def visible_or_404(
        context: AuthContext | None,
        library_id: str,
    ) -> dict:
        viewer_id = context.identity.id if context is not None else None
        item = repository.get_visible(viewer_id, library_id)
        if item is None:
            raise HTTPException(status_code=404, detail="library not found")
        return item

    def visible_version_or_404(
        context: AuthContext | None,
        library_id: str,
        branch_id: str,
        version_id: str,
        *,
        report: bool = False,
    ) -> dict:
        viewer_id = context.identity.id if context is not None else None
        item = repository.get_version_visible(
            viewer_id,
            library_id,
            branch_id,
            version_id,
        )
        if item is None:
            label = "report" if report else "version"
            raise HTTPException(status_code=404, detail=f"{label} not found")
        return item

    def publish_version(
        *,
        context: AuthContext,
        metadata: UploadMetadata,
        branch_key: str,
        source_run_id: str,
        known_library_id: str | None,
        known_branch_id: str | None,
        library_bytes: bytes,
        report_bytes: bytes,
        function_count: int,
        report_assets: list[tuple[str, bytes]] | None = None,
        compact_report_assets: bool = False,
    ) -> dict:
        store: ArtifactStore = application.state.artifact_store
        with application.state.write_lock:
            try:
                plan = repository.prepare_publication(
                    owner_id=context.identity.id,
                    model_id=metadata.model_id,
                    benchmarks=metadata.benchmarks,
                    branch_key=branch_key,
                    known_library_id=known_library_id,
                    known_branch_id=known_branch_id,
                    new_library_id=str(uuid.uuid4()),
                    new_branch_id=str(uuid.uuid4()),
                    data_identity=metadata.data_identity,
                )
            except SelectorNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except BranchIdentityConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            version_id = str(uuid.uuid4())
            version_prefix = (
                f"{context.identity.id}/{plan.library_id}/branches/{plan.branch_id}"
                f"/versions/{version_id}"
            )
            library_key = f"{version_prefix}/library.json"
            report_key = f"{version_prefix}/failure-card.html"
            created_at = _now()
            written_keys: list[str] = []
            try:
                store.put(library_key, library_bytes)
                written_keys.append(library_key)
                store.put(report_key, report_bytes)
                written_keys.append(report_key)
                asset_records: list[dict] = []
                archive_key = ""
                archive_writer = getattr(store, "put_archive", None)
                from .lazy_report import KEY_PREFIX, LazyReportAssets

                lazy_assets = isinstance(report_assets, LazyReportAssets)
                if ((compact_report_assets or lazy_assets) and report_assets
                        and callable(archive_writer)):
                    archive_key = f"{version_prefix}/report-assets.zip"
                    archive_writer(archive_key, report_assets)
                    written_keys.append(archive_key)
                for path, payload in report_assets or []:
                    if archive_key:
                        artifact_key = f"zip:{archive_key}!/{path}"
                    else:
                        artifact_key = f"{version_prefix}/report-assets/{path}"
                        store.put(artifact_key, payload)
                        written_keys.append(artifact_key)
                    # Only the in-process catalog can mint lazy descriptors. Uploads
                    # arrive as ordinary lists and cannot invoke local store reads.
                    if lazy_assets:
                        artifact_key = KEY_PREFIX + artifact_key
                    asset_records.append(
                        {
                            "path": path,
                            "artifact_key": artifact_key,
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                return repository.append_publication(
                    plan=plan,
                    identity=context.identity,
                    branch_key=branch_key,
                    source_run_id=source_run_id,
                    model_id=metadata.model_id,
                    benchmarks=metadata.benchmarks,
                    known_library_id=known_library_id,
                    known_branch_id=known_branch_id,
                    version_id=version_id,
                    created_at=created_at,
                    function_count=function_count,
                    library_sha256=hashlib.sha256(library_bytes).hexdigest(),
                    report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                    library_key=library_key,
                    report_key=report_key,
                    harness=metadata.harness,
                    source_harness=metadata.source_harness,
                    reference_data=[
                        reference.model_dump()
                        for reference in metadata.reference_data
                    ],
                    publish_trigger=metadata.publish_trigger,
                    report_assets=asset_records,
                    data_identity=metadata.data_identity,
                )
            except Exception as exc:
                logger.exception("publication append failed; deleting new artifacts")
                for key in reversed(written_keys):
                    try:
                        store.delete(key)
                    except Exception:
                        logger.exception("artifact cleanup failed for %s", key)
                if isinstance(exc, SelectorNotFoundError):
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                if isinstance(exc, BranchIdentityConflictError):
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                raise

    application.state.local_catalog = LocalCatalog(
        settings,
        database,
        lambda: application.state.artifact_store,
        lambda **kwargs: publish_version(
            context=AuthContext(identity=sessions.identity, method="local"),
            compact_report_assets=True,
            **kwargs,
        ),
        application.state.write_lock,
    )

    @application.get("/healthz", tags=["system"])
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @application.get(
        "/api/v1/me",
        response_model=MeResponse,
        tags=["identity"],
    )
    def api_me(context: AuthContext = require_auth_dependency) -> MeResponse:
        return MeResponse(
            **context.identity.model_dump(),
            auth_method=context.method,
        )

    @application.delete(
        "/api/v1/libraries/{library_id}",
        status_code=204,
        tags=["libraries"],
    )
    def api_delete_library(
        library_id: str,
        context: AuthContext = require_mutation_auth_dependency,
    ) -> Response:
        with application.state.write_lock:
            library = owned_or_404(context, library_id)
            if library["id"] != library_id:
                raise HTTPException(
                    status_code=409,
                    detail=("Historical library IDs cannot be used for deletion. "
                            "Open the current data card and delete using its current ID."),
                )
            store: ArtifactStore = application.state.artifact_store
            for key in repository.iter_artifact_keys(context.identity.id, library_id):
                with suppress(FileNotFoundError):
                    store.delete(key)
            if not repository.delete_owned(context.identity.id, library_id):
                raise RuntimeError("library disappeared during delete")
        return Response(status_code=204)

    @application.post(
        "/api/v2/publications",
        response_model=PublicationRecord,
        status_code=201,
        tags=["publications"],
    )
    async def api_create_publication(
        metadata: Annotated[str, Form()],
        library: Annotated[UploadFile, File()],
        failure_card: Annotated[UploadFile, File()],
        report_assets: Annotated[UploadFile | None, File()] = None,
        trace_references: Annotated[str | None, Form()] = None,
        context: AuthContext = require_mutation_auth_dependency,
    ) -> PublicationRecord:
        _require_content_type(
            library,
            {"application/json", "text/json", "application/octet-stream"},
            "library",
        )
        _require_content_type(
            failure_card,
            {"text/html", "application/xhtml+xml", "application/octet-stream"},
            "failure-card",
        )
        library_bytes = await _bounded_upload(library, settings.max_library_bytes, "library")
        report_bytes = await _bounded_upload(
            failure_card, settings.max_report_bytes, "failure-card"
        )
        parsed_report_assets: list[tuple[str, bytes]] = []
        if report_assets is not None:
            _require_content_type(
                report_assets,
                {
                    "application/zip",
                    "application/x-zip-compressed",
                    "application/octet-stream",
                },
                "failure-card assets",
            )
            archive_bytes = await _bounded_upload(
                report_assets,
                settings.max_report_assets_bytes,
                "failure-card assets",
            )
            parsed_report_assets = _parse_report_assets(
                settings,
                report_bytes,
                archive_bytes,
            )
        parsed_metadata, library_payload, function_count = _parse_publication_uploads(
            settings,
            metadata,
            library_bytes,
            report_bytes,
        )
        from .lazy_report import report_for_saved_run

        if trace_references is not None:
            from .trace_transfer import report_for_transferred_traces

            local_report = await run_in_threadpool(
                report_for_transferred_traces, settings, trace_references, library_payload,
                parsed_metadata.source_harness,
            )
        else:
            local_report = await run_in_threadpool(
                report_for_saved_run, settings, parsed_metadata.source_run_id, library_payload,
            )
        if local_report is not None:
            report_bytes = _report_with_saved_efficiency(local_report[0], report_bytes)
            parsed_report_assets = local_report[1]
        saved = await run_in_threadpool(
            publish_version,
            context=context,
            metadata=parsed_metadata,
            branch_key=parsed_metadata.branch_key,
            source_run_id=parsed_metadata.source_run_id,
            known_library_id=parsed_metadata.library_id,
            known_branch_id=parsed_metadata.branch_id,
            library_bytes=library_bytes,
            report_bytes=report_bytes,
            function_count=function_count,
            report_assets=parsed_report_assets,
        )
        return _to_publication_record(saved)

    @application.get(
        "/api/v2/libraries",
        response_model=LibraryPage,
        tags=["libraries"],
    )
    def api_v2_list_libraries(
        context: AuthContext = require_auth_dependency,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        model: str | None = None,
        benchmark: str | None = None,
    ) -> LibraryPage:
        items, total = repository.list_owned(
            context.identity.id,
            limit=limit,
            offset=offset,
            model_id=model,
            benchmark=benchmark,
        )
        return LibraryPage(
            items=[to_record(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/api/v2/libraries/{library_id}/branches",
        response_model=BranchPage,
        tags=["libraries"],
    )
    def api_v2_list_branches(
        library_id: str,
        context: AuthContext | None = optional_auth_dependency,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> BranchPage:
        visible_or_404(context, library_id)
        items, total = repository.list_branches(
            library_id,
            limit=limit,
            offset=offset,
            viewer_id=context.identity.id if context is not None else None,
        )
        if total == 0:
            visible_or_404(context, library_id)
        return BranchPage(
            items=[BranchRecord.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/api/v2/libraries/{library_id}/branches/{branch_id}/versions",
        response_model=VersionPage,
        tags=["libraries"],
    )
    def api_v2_list_versions(
        library_id: str,
        branch_id: str,
        context: AuthContext | None = optional_auth_dependency,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> VersionPage:
        library_id = repository.resolve_library_id(library_id, branch_id=branch_id)
        visible_or_404(context, library_id)
        if repository.get_branch(library_id, branch_id) is None:
            raise HTTPException(status_code=404, detail="branch not found")
        items, total = repository.list_versions(
            library_id,
            branch_id,
            limit=limit,
            offset=offset,
            viewer_id=context.identity.id if context is not None else None,
        )
        if total == 0:
            visible_or_404(
                context, repository.resolve_library_id(library_id, branch_id=branch_id),
            )
        return VersionPage(
            items=[_to_version_record(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def version_or_404(
        context: AuthContext | None,
        library_id: str,
        branch_id: str,
        version_id: str,
    ) -> dict:
        return visible_version_or_404(
            context,
            library_id,
            branch_id,
            version_id,
        )

    @application.get(
        "/api/v2/libraries/{library_id}/branches/{branch_id}/versions/{version_id}",
        response_model=VersionRecord,
        tags=["libraries"],
    )
    def api_v2_get_version(
        library_id: str,
        branch_id: str,
        version_id: str,
        context: AuthContext | None = optional_auth_dependency,
    ) -> VersionRecord:
        return _to_version_record(version_or_404(context, library_id, branch_id, version_id))

    @application.get(
        "/api/v2/libraries/{library_id}/branches/{branch_id}/versions/{version_id}/library.json",
        tags=["libraries"],
    )
    def api_v2_download_library(
        library_id: str,
        branch_id: str,
        version_id: str,
        context: AuthContext | None = optional_auth_dependency,
    ) -> Response:
        item = version_or_404(context, library_id, branch_id, version_id)
        try:
            content = application.state.artifact_store.get(item["library_key"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="library artifact not found") from exc
        is_public = bool(item["owner_is_public"] and item["is_public"])
        return Response(
            content,
            media_type="application/json",
            headers={
                "Cache-Control": (
                    "no-store" if is_public else "private, max-age=31536000, immutable"
                )
            },
        )

    @application.get(
        "/api/v2/libraries/{library_id}/branches/{branch_id}/versions/{version_id}/failure-card",
        tags=["libraries"],
    )
    def api_v2_download_report(
        request: Request,
        library_id: str,
        branch_id: str,
        version_id: str,
        context: AuthContext | None = optional_auth_dependency,
    ) -> Response:
        item = visible_version_or_404(
            context,
            library_id,
            branch_id,
            version_id,
            report=True,
        )
        is_public = bool(item["owner_is_public"] and item["is_public"])
        if _etag_matches(request, item["report_sha256"]):
            return _report_response(
                b"",
                immutable=True,
                public=is_public,
                etag=item["report_sha256"],
                status_code=304,
            )
        try:
            content = application.state.artifact_store.get(item["report_key"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report artifact not found") from exc
        return _report_response(
            content,
            immutable=True,
            public=is_public,
            etag=item["report_sha256"],
        )

    @application.get("/api/openapi.json", include_in_schema=False)
    def authenticated_openapi(
        _context: AuthContext = require_auth_dependency,
    ) -> JSONResponse:
        return JSONResponse(application.openapi())

    @application.get("/api/docs", include_in_schema=False)
    def authenticated_docs(
        _context: AuthContext = require_auth_dependency,
    ) -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url="/api/openapi.json",
            title="AutoFA Local Library API",
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request) -> Response:
        context = sessions.from_request(request)
        items, _ = repository.list_visible(
            context.identity.id,
            limit=1000,
            offset=0,
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "me": context.identity,
                "libraries": items,
                "catalog_indexing": application.state.local_catalog.indexing,
                "pending_reports": application.state.local_catalog.pending_reports(),
            },
        )

    def resolve_dashboard_selection(
        context: AuthContext | None,
        library_id: str,
        branch_id: str | None,
        version_id: str | None,
    ) -> tuple[dict, list[dict], dict, list[dict], dict]:
        library_id = repository.resolve_library_id(library_id, branch_id=branch_id)
        library = visible_or_404(context, library_id)
        branches, _ = repository.list_branches(
            library_id,
            limit=1_000_000,
            offset=0,
            viewer_id=context.identity.id if context is not None else None,
        )
        selected_branch_id = branch_id or library["latest_branch_id"]
        selected_branch = next(
            (branch for branch in branches if branch["id"] == selected_branch_id), None,
        )
        if selected_branch is None:
            raise HTTPException(status_code=404, detail="branch not found")
        versions, _ = repository.list_versions(
            library_id,
            selected_branch_id,
            limit=1_000_000,
            offset=0,
            viewer_id=context.identity.id if context is not None else None,
        )
        selected_version_id = version_id or (
            library["latest_version_id"]
            if selected_branch_id == library["latest_branch_id"]
            else selected_branch["latest_version_id"]
        )
        selected_version = visible_version_or_404(
            context,
            library_id,
            selected_branch_id,
            selected_version_id,
        )
        return library, branches, selected_branch, versions, selected_version

    @application.get(
        "/libraries/{library_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def library_detail(
        request: Request,
        library_id: str,
        branch_id: str | None = None,
        version_id: str | None = None,
        failure: str | None = None,
    ) -> Response:
        context = sessions.from_request(request)
        library, branches, selected_branch, versions, selected_version = (
            resolve_dashboard_selection(
                context,
                library_id,
                branch_id,
                version_id,
            )
        )
        is_owner = context is not None and context.identity.id == library["owner_id"]
        report_query = urlencode(
            {
                "branch_id": selected_branch["id"],
                "version_id": selected_version["id"],
                "view": REPORT_VIEW_VERSION,
                **({"failure": failure} if failure else {}),
            }
        )
        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "me": context.identity if context is not None else None,
                "csrf_token": context.csrf_token if context is not None else None,
                "is_owner": is_owner,
                "library": library,
                "merge_candidates": [
                    item for item in repository.list_owned(
                        context.identity.id, limit=1_000_000, offset=0,
                    )[0] if item["id"] != library["id"]
                ] if is_owner else [],
                "branches": branches,
                "selected_branch": selected_branch,
                "versions": versions,
                "selected_version": selected_version,
                "canonical_url": (
                    f"{settings.public_url.rstrip('/')}/libraries/{library_id}?{report_query}"
                ),
                "report_query": report_query,
                "selected_failure": failure,
                "data_api_base_url": settings.data_api_base_url,
            },
            headers={"Cache-Control": "no-store"},
        )

    @application.get(
        "/libraries/{library_id}/report",
        include_in_schema=False,
    )
    def dashboard_report(
        request: Request,
        library_id: str,
        branch_id: str | None = None,
        version_id: str | None = None,
        failure: str | None = None,
        standalone: bool = False,
    ) -> Response:
        context = sessions.from_request(request)
        library, _, _, _, selected_version = resolve_dashboard_selection(
            context,
            library_id,
            branch_id,
            version_id,
        )
        is_owner = context is not None and context.identity.id == library["owner_id"]
        is_public = bool(library["owner_is_public"] and library["is_public"])
        if not is_owner and not is_public:
            raise HTTPException(status_code=404, detail="report not found")
        if standalone:
            return templates.TemplateResponse(request, "report_fullscreen.html", {
                "report_url": str(request.url.remove_query_params("standalone")
                                   .include_query_params(theme=ui_theme(request))),
                "selected_failure": failure,
                "data_api_base_url": settings.data_api_base_url,
            }, headers={"Cache-Control": "no-store"})
        immutable = branch_id is not None and version_id is not None
        asset_token = report_asset_token(selected_version["id"])
        display_etag = hashlib.sha256(
            (
                f"{selected_version['report_sha256']}:{asset_token}:{failure or ''}"
                f":report-view-{REPORT_VIEW_VERSION}"
            ).encode()
        ).hexdigest()
        if immutable and _etag_matches(request, display_etag):
            return _report_response(
                b"",
                public=is_public,
                etag=display_etag,
                status_code=304,
            )
        try:
            content = application.state.artifact_store.get(selected_version["report_key"])
            library_content = application.state.artifact_store.get(selected_version["library_key"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report artifact not found") from exc
        asset_prefix = f"/report-assets/{selected_version['id']}/{asset_token}"
        content = _report_with_current_template(content, library_content)
        content = _report_with_assets_base(content, asset_prefix)
        return _report_response(
            content,
            public=is_public,
            etag=display_etag if immutable else None,
        )

    @application.get(
        "/report-assets/{version_id}/{access_token}/{asset_path:path}",
        include_in_schema=False,
    )
    def dashboard_report_asset(
        version_id: str,
        access_token: str,
        asset_path: str,
    ) -> Response:
        expected_token = report_asset_token(version_id)
        if not secrets.compare_digest(access_token, expected_token):
            raise HTTPException(status_code=404, detail="report asset not found")
        safe_path = _safe_report_asset_path(asset_path)
        item = repository.get_report_asset(version_id, safe_path)
        if item is None:
            raise HTTPException(status_code=404, detail="report asset not found")
        try:
            content = application.state.artifact_store.get(item["artifact_key"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report asset not found") from exc
        return _report_asset_response(content)

    def merge_cards(context: AuthContext, library_id: str, target_library_id: str) -> dict:
        with application.state.write_lock:
            try:
                return repository.merge_libraries(
                    owner_id=context.identity.id,
                    source_library_id=library_id,
                    target_library_id=target_library_id,
                )
            except SelectorNotFoundError as exc:
                raise HTTPException(status_code=404, detail="library not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post(
        "/api/v2/libraries/{library_id}/merge",
        response_model=LibraryRecord,
        tags=["libraries"],
    )
    def api_merge_libraries(
        library_id: str,
        target_library_id: Annotated[str, Body(embed=True, min_length=1, max_length=100)],
        context: AuthContext = require_mutation_auth_dependency,
    ) -> LibraryRecord:
        return to_record(merge_cards(context, library_id, target_library_id))

    @application.post("/libraries/{library_id}/merge", include_in_schema=False)
    async def dashboard_merge_libraries(
        request: Request,
        library_id: str,
        target_library_id: Annotated[str, Form(min_length=1, max_length=100)],
        csrf_token: Annotated[str, Form()],
    ) -> Response:
        context = browser_context(request)
        if not secrets.compare_digest(csrf_token, context.csrf_token):
            raise HTTPException(status_code=403, detail="CSRF validation failed")
        merged = merge_cards(context, library_id, target_library_id)
        return RedirectResponse(f"/libraries/{merged['id']}", status_code=303)

    @application.post(
        "/libraries/{library_id}/delete",
        include_in_schema=False,
    )
    async def dashboard_delete(
        request: Request,
        library_id: str,
        csrf_token: Annotated[str, Form()],
    ) -> Response:
        context = browser_context(request)
        if not secrets.compare_digest(csrf_token, context.csrf_token):
            raise HTTPException(status_code=403, detail="CSRF validation failed")
        api_delete_library(library_id, context)
        return RedirectResponse("/", status_code=303)

    return application
