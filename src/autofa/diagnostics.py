# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured diagnostics — stop silent degradations.

Events are aggregated by ``code`` so 50k truncation notices become one entry
with a count. Persisted under ``run_dir()/diagnostics.jsonl``; Python logging
is also wired to ``run_dir()/autofa.log``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("autofa")

# Severity levels (ordered).
CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_MAX_AGGREGATED = 500


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    remediation: str = ""
    count: int = 1
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Diagnostics:
    """Process-wide collector; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_code: OrderedDict[str, Diagnostic] = OrderedDict()
        self._path: Path | None = None
        self._log_configured = False

    def configure(self, run_dir: Path) -> None:
        """Attach file sinks under ``run_dir``."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._path = run_dir / "diagnostics.jsonl"
            if not self._log_configured:
                _configure_logging(run_dir / "autofa.log")
                self._log_configured = True

    @contextmanager
    def configured(self, run_dir: Path, *, console: bool = False) -> Iterator[None]:
        """Temporarily route diagnostics and logs to one run directory."""
        root = logging.getLogger("autofa")
        with self._lock:
            previous_path = self._path
            previous_level = root.level
            previous_handlers = [
                handler
                for handler in root.handlers
                if getattr(handler, "_autofa_file", False)
                or getattr(handler, "_autofa_console", False)
            ]
            for handler in previous_handlers:
                root.removeHandler(handler)
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._path = run_dir / "diagnostics.jsonl"
            _configure_logging(run_dir / "autofa.log", console=console)
        try:
            yield
        finally:
            with self._lock:
                for handler in list(root.handlers):
                    if getattr(handler, "_autofa_file", False) or getattr(
                        handler, "_autofa_console", False
                    ):
                        root.removeHandler(handler)
                        handler.close()
                for handler in previous_handlers:
                    root.addHandler(handler)
                root.setLevel(previous_level)
                self._path = previous_path

    def record(
        self,
        code: str,
        *,
        severity: str = WARNING,
        message: str,
        context: dict[str, Any] | None = None,
        remediation: str = "",
    ) -> Diagnostic:
        now = datetime.now(UTC).isoformat()
        ctx = dict(context or {})
        with self._lock:
            existing = self._by_code.get(code)
            if existing is not None:
                existing.count += 1
                existing.last_seen = now
                # Keep the richest message / remediation.
                if len(message) > len(existing.message):
                    existing.message = message
                if remediation and not existing.remediation:
                    existing.remediation = remediation
                if ctx:
                    existing.context.update(
                        {k: v for k, v in ctx.items() if k not in existing.context}
                    )
                diag = existing
            else:
                if len(self._by_code) >= _MAX_AGGREGATED:
                    self._by_code.popitem(last=False)
                diag = Diagnostic(
                    code=code,
                    severity=severity,
                    message=message,
                    context=ctx,
                    remediation=remediation,
                    count=1,
                    first_seen=now,
                    last_seen=now,
                )
                self._by_code[code] = diag
            path = self._path
        if path is not None:
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(diag.to_dict(), default=str) + "\n")
            except OSError:  # pragma: no cover
                log.debug("could not write diagnostics to %s", path, exc_info=True)
        level = {
            CRITICAL: logging.ERROR,
            WARNING: logging.WARNING,
            INFO: logging.INFO,
        }.get(severity, logging.WARNING)
        log.log(level, "[%s] %s", code, message)
        return diag

    def critical(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        remediation: str = "",
    ) -> Diagnostic:
        return self.record(
            code, severity=CRITICAL, message=message, context=context, remediation=remediation
        )

    def warning(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        remediation: str = "",
    ) -> Diagnostic:
        return self.record(
            code, severity=WARNING, message=message, context=context, remediation=remediation
        )

    def info(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        remediation: str = "",
    ) -> Diagnostic:
        return self.record(
            code, severity=INFO, message=message, context=context, remediation=remediation
        )

    def list(self) -> list[Diagnostic]:
        with self._lock:
            items = list(self._by_code.values())
        order = {CRITICAL: 0, WARNING: 1, INFO: 2}
        return sorted(items, key=lambda d: (order.get(d.severity, 9), -d.count, d.code))

    def summary(self) -> dict[str, Any]:
        items = self.list()
        return {
            "n": len(items),
            "n_critical": sum(1 for d in items if d.severity == CRITICAL),
            "n_warning": sum(1 for d in items if d.severity == WARNING),
            "n_info": sum(1 for d in items if d.severity == INFO),
            "items": [d.to_dict() for d in items],
            "log_path": str(self._path) if self._path else None,
        }

    def preflight_lines(self) -> list[str]:
        """Human-readable lines for the CLI preflight panel."""
        lines: list[str] = []
        for d in self.list():
            if d.severity == INFO and d.count == 1 and not d.remediation:
                continue
            prefix = {"critical": "CRIT", "warning": "WARN", "info": "INFO"}.get(
                d.severity, d.severity.upper()
            )
            count = f" ×{d.count}" if d.count > 1 else ""
            line = f"[{prefix}{count}] {d.message}"
            if d.remediation:
                line += f" → {d.remediation}"
            lines.append(line)
        return lines


_DIAGNOSTICS = Diagnostics()


def get_diagnostics() -> Diagnostics:
    return _DIAGNOSTICS


def _configure_logging(log_path: Path, *, console: bool = False) -> None:
    root = logging.getLogger("autofa")
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for h in list(root.handlers):
        if getattr(h, "_autofa_file", False) or getattr(h, "_autofa_console", False):
            root.removeHandler(h)
            h.close()
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        fh._autofa_file = True  # type: ignore[attr-defined]
        root.addHandler(fh)
    except OSError:  # pragma: no cover
        pass
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler._autofa_console = True  # type: ignore[attr-defined]
        root.addHandler(console_handler)
