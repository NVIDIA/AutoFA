# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile identity and token-gated browser sessions for a shared server."""
from __future__ import annotations

import hashlib
import hmac
import html
import secrets
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from fastapi import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .config import Settings
from .schemas import Identity

_COOKIE = "autofa_library_session"
_SESSION_SECONDS = 7 * 24 * 60 * 60
_LOGIN_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "same-origin",
}


@dataclass(frozen=True, slots=True)
class AuthContext:
    identity: Identity
    method: str
    csrf_token: str = ""


def safe_next_url(value: str | None) -> str:
    value = (value or "/").strip()
    parsed = urlsplit(value)
    if not value.startswith("/") or value.startswith("//") or "\\" in value or parsed.netloc:
        return "/"
    return urlunsplit(("", "", parsed.path or "/", parsed.query, ""))


class SessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.identity = Identity(id=1, username="local", name="Local researcher")
        # Rotating the shared token invalidates existing browser sessions, even
        # when a separate session secret is configured.
        self._key = hashlib.sha256(
            f"autofa-browser:{settings.local_token}:{settings.session_secret}".encode()
        ).digest()

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def _cookie_context(self, cookie: str) -> AuthContext | None:
        if not cookie or len(cookie) > 256:
            return None
        try:
            expires, nonce, signature = cookie.split(".")
            deadline = int(expires)
        except (ValueError, TypeError):
            return None
        if (not nonce or deadline <= time.time()
                or deadline > time.time() + _SESSION_SECONDS + 60
                or not secrets.compare_digest(
                    signature.encode(), self._sign(f"{expires}.{nonce}").encode(),
                )):
            return None
        return AuthContext(identity=self.identity, method="session",
                           csrf_token=self._sign(f"csrf:{cookie}"))

    def from_request(self, request: Request) -> AuthContext | None:
        if not self.settings.remote_hosts:
            return AuthContext(identity=self.identity, method="session",
                               csrf_token=self.settings.local_token)
        authorization = request.headers.get("authorization", "")
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer" and secrets.compare_digest(
                token.encode(), self.settings.local_token.encode(),
            ):
                return AuthContext(identity=self.identity, method="local_token")
            return None
        return self._cookie_context(request.cookies.get(_COOKIE, ""))

    def protect_request(self, request: Request) -> Response | None:
        """Gate the entire remote server, including native trace and static routes.

        Report sidecars already have version-bound capability tokens. Sandboxed
        reports have an opaque origin and cannot rely on the browser cookie.
        The route itself verifies that token before returning any bytes.
        """
        if not self.settings.remote_hosts or self.from_request(request) is not None:
            return None
        path = request.url.path
        if path in {"/login", "/healthz"}:
            return None
        if request.method in {"GET", "HEAD"} and path.startswith("/report-assets/"):
            return None
        if (request.method == "GET" and not path.startswith(("/api/", "/static/"))
                and "/api/" not in path and not path.startswith("/trace-static/")):
            target = safe_next_url(path + (f"?{request.url.query}" if request.url.query else ""))
            return RedirectResponse("/login?" + urlencode({"next": target}), status_code=303,
                                    headers={"Cache-Control": "no-store"})
        return JSONResponse({"detail": "Library server authentication required"}, status_code=401,
                            headers={"Cache-Control": "no-store"})

    def _login_page(self, target: str, *, error: str = "", status_code: int = 200) -> Response:
        error_html = f'<p role="alert">{html.escape(error)}</p>' if error else ""
        return HTMLResponse(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Sign in to AutoFA Library Server</title>'
            '<style>body{font:16px system-ui;max-width:34rem;margin:12vh auto;padding:1rem;'
            'color-scheme:light dark}label{display:block;margin:1rem 0 .4rem}'
            'input{font:inherit;padding:.6rem;box-sizing:border-box;width:100%}'
            'button{font:inherit;padding:.6rem 1rem;margin-top:1rem}'
            '[role=alert]{color:#b33}</style><h1>AutoFA Library Server</h1>'
            '<p>Enter the shared server token to view reports and traces.</p>'
            + error_html + '<form action="/login" method="post">'
            f'<input type="hidden" name="next" value="{html.escape(target, quote=True)}">'
            '<label for="token">Server token</label>'
            '<input id="token" name="token" type="password" required '
            'autocomplete="current-password" maxlength="4096">'
            '<button type="submit">Sign in</button></form></html>',
            status_code=status_code, headers=_LOGIN_HEADERS,
        )

    def mount_routes(self, application) -> None:
        @application.get("/login", include_in_schema=False)
        def login_page(request: Request) -> Response:
            target = safe_next_url(request.query_params.get("next"))
            if self.from_request(request) is not None:
                return RedirectResponse(target, status_code=303)
            return self._login_page(target)

        @application.post("/login", include_in_schema=False)
        async def login(request: Request) -> Response:
            # This route runs behind the same Host/Origin checks as publication.
            # Bound the body before parsing; a password must never be reflected.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 8192:
                    return JSONResponse({"detail": "Login request is too large"}, status_code=413)
            try:
                fields = parse_qs(body.decode("utf-8"), max_num_fields=4)
            except (UnicodeError, ValueError):
                return self._login_page("/", error="Invalid sign-in request", status_code=400)
            target = safe_next_url(fields.get("next", ["/"])[0])
            token = fields.get("token", [""])[0]
            if not token or not secrets.compare_digest(
                token.encode(), self.settings.local_token.encode(),
            ):
                return self._login_page(target, error="Invalid server token", status_code=401)
            payload = f"{int(time.time()) + _SESSION_SECONDS}.{secrets.token_hex(16)}"
            response = RedirectResponse(target, status_code=303, headers=_LOGIN_HEADERS)
            response.set_cookie(
                _COOKIE, f"{payload}.{self._sign(payload)}", max_age=_SESSION_SECONDS,
                httponly=True, secure=request.url.scheme == "https", samesite="strict", path="/",
            )
            return response

        @application.post("/logout", include_in_schema=False)
        def logout(request: Request) -> Response:
            context = self.from_request(request)
            if context is None:
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
            if context.method == "session" and not secrets.compare_digest(
                request.headers.get("X-CSRF-Token", "").encode(), context.csrf_token.encode(),
            ):
                return JSONResponse({"detail": "CSRF validation failed"}, status_code=403)
            response = RedirectResponse("/login", status_code=303, headers=_LOGIN_HEADERS)
            response.delete_cookie(_COOKIE, path="/")
            return response
