# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running prompt_toolkit apps from sync or async callers."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from prompt_toolkit.patch_stdout import patch_stdout

if TYPE_CHECKING:
    from prompt_toolkit.application import Application


def run_app(app: Application) -> None:
    """Run a prompt_toolkit ``Application`` whether or not an event loop is active.

    Claude session chat already owns an asyncio loop. Recent prompt_toolkit
    ``Application.run()`` uses ``asyncio.run()`` internally, which raises
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    (and leaves ``run_async`` un-awaited). When a loop is already running,
    execute ``app.run()`` on a worker thread with its own loop so the TUI
    can take over stdin until it exits.
    """
    def run() -> None:
        # Background server logs and other output must suspend/redraw the TUI
        # instead of moving its cursor behind the renderer's back. Create the
        # proxy in the app's thread so it follows that thread's AppSession.
        with patch_stdout(raw=True):
            app.run()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        run()
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(run).result()
