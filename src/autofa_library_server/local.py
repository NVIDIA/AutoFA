# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the library dashboard on this machine."""

from __future__ import annotations

import asyncio
import errno
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.parse import urlsplit

import uvicorn

from autofa.local_service import service_token, validate_service_url

if TYPE_CHECKING:
    from autofa.data.remote_library_api import RemoteLibraryClient


def _wait_for_dashboard(
    client: RemoteLibraryClient,
    *,
    deadline: float,
    check_worker: Callable[[], None] | None = None,
) -> None:
    """Wait for authenticated HTTP readiness within the startup budget."""
    from autofa.data.remote_library_api import RemoteLibraryError

    last_error: OSError | RemoteLibraryError | None = None
    timeout_message = f"Dashboard startup timed out at {client.base_url}"
    while True:
        if check_worker is not None:
            check_worker()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(timeout_message) from last_error
        client.timeout = min(0.5, remaining)
        try:
            identity = client.me()
        except (OSError, RemoteLibraryError) as error:
            # Read timeouts escape urllib as OSError; connection failures are
            # wrapped in RemoteLibraryError. HTTP and protocol errors cannot
            # establish readiness by retrying the same credentials.
            if isinstance(error, RemoteLibraryError) and (
                error.status is not None or not isinstance(error.__cause__, URLError)
            ):
                raise
            last_error = error
        else:
            if identity.auth_method != "local_token":
                raise RuntimeError(f"Dashboard authentication failed at {client.base_url}")
            if check_worker is not None:
                check_worker()
            if time.monotonic() >= deadline:
                raise RuntimeError(timeout_message)
            return
        if check_worker is not None:
            check_worker()
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.05, remaining))


def _request_catalog_stop(application) -> None:
    catalog = getattr(getattr(application, "state", None), "local_catalog", None)
    if catalog is not None:
        catalog.request_stop()


class _CatalogServer(uvicorn.Server):
    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Uvicorn calls lifespan cleanup after draining HTTP requests. Stop
        # background indexing first, for both CLI-owned and standalone servers.
        _request_catalog_stop(self.config.app)
        await super().shutdown(sockets=sockets)


@contextmanager
def managed_dashboard(*, url: str, startup_timeout: float = 10.0) -> Iterator[bool]:
    """Start a dashboard for this CLI, or reuse one belonging to this profile.

    Yield whether we own the server. A reused server keeps its original lifetime;
    only the CLI that started a server shuts it down. A daemon thread keeps the
    server in this process, including when the terminal or process is killed.
    """
    from autofa.data.remote_library_api import RemoteLibraryClient

    url = validate_service_url(url)
    parsed = urlsplit(url)
    host = "::1" if parsed.hostname == "::1" else "127.0.0.1"
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Reserve the port before creating a worker so bind errors are
            # handled by the CLI and never terminate a Uvicorn worker silently.
            sock.bind((host, parsed.port or 80))
            sock.listen(128)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            try:
                client = RemoteLibraryClient(
                    url,
                    token=service_token(),
                    credential_source="local",
                    timeout=0.5,
                )
                deadline = time.monotonic() + startup_timeout
                _wait_for_dashboard(client, deadline=deadline)
            except Exception as cause:
                raise RuntimeError(
                    f"Port in use at {url}, but no dashboard for this AUTOFA_HOME "
                    "could be reached. Set a different remote_library_base_url "
                    "in autofa.toml or stop the other listener."
                ) from cause
            yield False
            return

        from .config import Settings
        from .main import create_app

        settings = Settings(public_url=url)
        application = create_app(settings)
        server = _CatalogServer(
            uvicorn.Config(
                application,
                log_config=None,
                log_level="warning",
                loop="asyncio",
                access_log=False,
                timeout_graceful_shutdown=3,
            )
        )
        failures: list[BaseException] = []
        running: list[tuple[asyncio.AbstractEventLoop, asyncio.Task]] = []

        async def run_server() -> None:
            running.append((asyncio.get_running_loop(), asyncio.current_task()))
            await server.serve(sockets=[sock])

        def run() -> None:
            try:
                asyncio.run(run_server())
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=run, name="autofa-local-dashboard", daemon=True)

        def check_worker() -> None:
            if not thread.is_alive():
                raise RuntimeError(f"Dashboard startup failed at {url}") from (
                    failures[0] if failures else None
                )

        deadline = time.monotonic() + startup_timeout
        thread.start()
        try:
            while not server.started:
                check_worker()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Dashboard startup timed out at {url}")
                time.sleep(min(0.01, remaining))
            # Check the same URL the CLI uses, including localhost resolution.
            client = RemoteLibraryClient(
                url,
                token=settings.local_token,
                credential_source="local",
                timeout=0.5,
            )
            _wait_for_dashboard(client, deadline=deadline, check_worker=check_worker)
            yield True
        finally:
            # CLI finalization/publication finishes before this context exits.
            # Stop scheduling catalog work before Uvicorn drains HTTP requests;
            # lifespan cleanup happens only after that drain has completed.
            _request_catalog_stop(application)
            server.should_exit = True
            thread.join(timeout=5)
            if thread.is_alive():
                server.force_exit = True
                if running:
                    loop, task = running[0]
                    try:
                        loop.call_soon_threadsafe(task.cancel)
                    except RuntimeError:
                        pass  # The worker finished and closed its loop meanwhile.
                thread.join(timeout=1)


def serve(*, port: int = 8765, host: str = "127.0.0.1") -> None:
    if not 1 <= port <= 65535:
        raise ValueError("Dashboard port must be between 1 and 65535")
    from .config import Settings
    from .main import create_app

    url = f"http://127.0.0.1:{port}"
    remote = host not in {"127.0.0.1", "localhost", "::1"}
    settings = Settings(public_url=url, remote_hosts=remote)
    print(f"AutoFA Local Library Server: {url}\nData: {settings.data_dir}")
    if remote:
        print(f"Listening on {host}:{port}. Browser access and publication require the server "
              "token.\n"
              f"Token file: {settings.data_dir / 'token'}\n"
              "On another AutoFA machine, open Settings > Library server, choose AutoFA server, "
              "and set AUTOFA_LIBRARY_TOKEN to this token.")
    server = _CatalogServer(uvicorn.Config(create_app(settings), host=host, port=port))
    try:
        server.run()
    except KeyboardInterrupt:
        pass  # Preserve uvicorn.run's clean Ctrl+C exit after server cleanup.
