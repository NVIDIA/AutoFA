# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The colour scheme a page is drawn in: the failure-card report's preference, which theme.js
mirrors into a cookie so the server can draw the page, and the reports it embeds, in it."""

from __future__ import annotations

from starlette.requests import Request

COOKIE = "autofa_theme"


def ui_theme(request: Request) -> str:
    return "dark" if request.cookies.get(COOKIE) == "dark" else "light"
